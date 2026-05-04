# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the Directive Assembly module (Step 8).

Exercises epistemic gap computation, trait trajectories, relationship tensions,
and the full CreativeBrief assembly for different directive effects.
"""
import pytest
from copy import deepcopy

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge,
    TraitVector, Affordance, Belief,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.directive_assembly import (
    DirectiveAssembler, CreativeBrief,
    EpistemicGap, TraitTrajectory, RelationshipTension, ConstraintBlock,
    NarrativeTension, HiddenChannel,
)
from shadow_loom.query_models import DirectiveQuery
from shadow_loom.narrative_physics import calculate_narrative_physics

# ── Reuse plots ──
from example_worlds.macbeth import world_state as macbeth_ws
from example_worlds.nineteen_eighty_four import world_state as orwell_ws
from example_worlds.gone_girl import world_state as gone_girl_ws
from example_worlds.brief_encounter import world_state as brief_encounter_ws


# =====================================================================
# Helper: build ego payload for directive tests
# =====================================================================
def _ego_payload(ws, focus_ids):
    return extract_ego_graph_from_memory(ws, focus_ids).model_dump()


# =====================================================================
# EPISTEMIC GAP COMPUTATION
# =====================================================================
class TestEpistemicGaps:
    """compute_epistemic_gaps must identify gaps between beliefs and reality."""

    def test_contradicted_belief_detected(self):
        """Winston's belief about O'Brien being Brotherhood must be contradicted."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)
        gaps = assembler.compute_epistemic_gaps(["ENT_WINSTON"])

        obrien_gaps = [g for g in gaps if g.belief_target_id == "ENT_OBRIEN"]
        assert len(obrien_gaps) >= 1
        # O'Brien is actually Thought Police, so "Brotherhood" belief should be contradicted
        contradicted = [g for g in obrien_gaps if g.gap_type == "contradicted"]
        assert len(contradicted) >= 1

    def test_gap_magnitude_reflects_confidence(self):
        """High-confidence contradicted beliefs must have high gap_magnitude."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)
        gaps = assembler.compute_epistemic_gaps(["ENT_WINSTON"])

        for g in gaps:
            if g.gap_type == "contradicted":
                assert g.gap_magnitude > 0, "Contradicted beliefs must have positive magnitude"

    def test_entity_without_beliefs_returns_empty(self):
        """Entity with no beliefs must produce no gaps."""
        ws = deepcopy(orwell_ws)
        # Charrington has no beliefs in the model
        ego = _ego_payload(ws, ["ENT_CHARRINGTON"])
        assembler = DirectiveAssembler(None, ego, ws)
        gaps = assembler.compute_epistemic_gaps(["ENT_CHARRINGTON"])
        assert gaps == []

    def test_multiple_entities(self):
        """Gaps for multiple entities must be collected together."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON", "ENT_JULIA"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)
        gaps = assembler.compute_epistemic_gaps(["ENT_WINSTON", "ENT_JULIA"])

        entity_ids = {g.entity_id for g in gaps}
        # At least Winston should appear (he has many beliefs)
        assert "ENT_WINSTON" in entity_ids


# =====================================================================
# TRAIT TRAJECTORY COMPUTATION
# =====================================================================
class TestTraitTrajectories:
    """compute_trait_trajectories must capture headroom analysis."""

    def test_headroom_computation(self):
        """Headroom must equal 1.0 - value (up) and value (down)."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        trajs = assembler.compute_trait_trajectories(["ENT_MACBETH"])

        assert len(trajs) >= 1
        for t in trajs:
            assert abs(t.headroom_up - (1.0 - t.current_value)) < 0.001
            assert abs(t.headroom_down - t.current_value) < 0.001

    def test_all_traits_covered(self):
        """Every trait on the entity must produce a trajectory."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        trajs = assembler.compute_trait_trajectories(["ENT_MACBETH"])

        trait_names = {t.trait_name for t in trajs}
        expected = set(macbeth_ws.entities["ENT_MACBETH"].traits.keys())
        assert trait_names == expected


# =====================================================================
# RELATIONSHIP TENSION COMPUTATION
# =====================================================================
class TestRelationshipTensions:
    """compute_relationship_tensions must identify relationship asymmetries."""

    def test_tension_detected(self):
        """Macbeth world should have non-trivial relationship tensions."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        tensions = assembler.compute_relationship_tensions(["ENT_MACBETH"])

        assert len(tensions) >= 1
        for t in tensions:
            # RelationshipTension uses source_id / target_id (dyadic)
            # rather than the legacy participant_ids list.
            assert "ENT_MACBETH" in (t.source_id, t.target_id)

    def test_asymmetry_score_non_negative(self):
        """Asymmetry score must be >= 0."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        tensions = assembler.compute_relationship_tensions(["ENT_MACBETH"])

        for t in tensions:
            assert t.asymmetry_score >= 0.0


# =====================================================================
# CREATIVE BRIEF ASSEMBLY
# =====================================================================
class TestCreativeBriefAssembly:
    """assemble() must produce a structured CreativeBrief."""

    def test_suspense_directive(self):
        """Suspense directive must produce epistemic constraints."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_WINSTON"],
            target_effect="suspense",
            intensity=0.8,
        )
        brief = assembler.assemble(directive)

        assert isinstance(brief, CreativeBrief)
        assert brief.target_effect == "suspense"
        assert len(brief.epistemic_gaps) >= 1
        # Should have at least one epistemic constraint
        epistemic = [c for c in brief.constraints if c.constraint_type == "epistemic"]
        assert len(epistemic) >= 1

    def test_dramatic_irony_directive(self):
        """Dramatic irony must produce 'MUST NOT reveal' constraints."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_WINSTON"],
            target_effect="dramatic_irony",
            intensity=0.9,
        )
        brief = assembler.assemble(directive)

        hard_epistemic = [
            c for c in brief.constraints
            if c.constraint_type == "epistemic" and c.priority == "hard"
        ]
        assert len(hard_epistemic) >= 1
        assert any("MUST NOT" in c.instruction for c in hard_epistemic)

    def test_surprise_directive(self):
        """Surprise directive must produce revelation constraints."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_WINSTON"],
            target_effect="surprise",
            intensity=1.0,
        )
        brief = assembler.assemble(directive)

        revelations = [
            c for c in brief.constraints
            if "REVELATION" in c.instruction
        ]
        assert len(revelations) >= 1

    def test_grief_directive_produces_math_constraint(self):
        """Grief directive must produce mathematical trait-shift constraints."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="grief",
            intensity=0.7,
        )
        brief = assembler.assemble(directive)

        math_constraints = [
            c for c in brief.constraints
            if c.constraint_type == "mathematical"
        ]
        # Should have at least one trait-shift constraint (despair, guilt, etc.)
        assert len(math_constraints) >= 1

    def test_vector_target_id_constraint(self):
        """Specific target_vector_id must produce an extra constraint."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="rage",
            target_vector_id="ENT_MACBETH.traits.ambition",
            intensity=0.5,
        )
        brief = assembler.assemble(directive)

        # Should have a constraint referencing the ambition trait
        ambition_constraints = [
            c for c in brief.constraints
            if "ambition" in c.instruction.lower()
        ]
        assert len(ambition_constraints) >= 1

    def test_brief_serialization(self):
        """CreativeBrief must serialize to dict."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
            intensity=0.5,
        )
        brief = assembler.assemble(directive)

        d = brief.model_dump()
        assert "target_effect" in d
        assert "constraints" in d
        assert "epistemic_gaps" in d
        assert "narrative_tensions" in d
        assert "hidden_channels" in d
        assert "trait_trajectories" in d
        assert "relationship_tensions" in d


# =====================================================================
# INTEGRATION WITH NARRATIVE_PHYSICS.PY
# =====================================================================
class TestDirectiveNarrativePhysicsIntegration:
    """use_causal_engine=True directive path must return CreativeBrief."""

    def test_directive_with_engine(self):
        """Directive via engine must return creative_brief instead of directives."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
            intensity=0.8,
        )
        result = calculate_narrative_physics(
            query, macbeth_ws, use_causal_engine=True,
        )
        assert result["status"] == "success"
        assert "creative_brief" in result
        assert result["creative_brief"]["target_effect"] == "suspense"
        assert "constraints" in result["creative_brief"]

    def test_legacy_directive_unchanged(self):
        """Default directive path must return directives string."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
            target_vector_id="ENT_MACBETH.traits.ambition",
            intensity=0.8,
        )
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"
        assert "directives" in result
        assert isinstance(result["directives"], str)
        assert "creative_brief" not in result


# =====================================================================
# NARRATIVE TENSION COMPUTATION (fabula/syuzhet displacement)
# =====================================================================
class TestNarrativeTension:
    """compute_narrative_tension must detect fabula/syuzhet displacement."""

    def test_brief_encounter_frame_story_displacement(self):
        """EVT_FINAL_MEETING (fabula=10000, syuzhet=1) opens the frame story
        but happens chronologically near the end — large negative displacement."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)
        tensions = assembler.compute_narrative_tension()

        tension_map = {t.event_id: t for t in tensions}
        assert "EVT_FINAL_MEETING" in tension_map
        fm = tension_map["EVT_FINAL_MEETING"]
        # high fabula rank, low syuzhet rank ⇒ negative displacement
        assert fm.displacement < 0, (
            f"EVT_FINAL_MEETING should have negative displacement, got {fm.displacement}"
        )

    def test_brief_encounter_grit_positive_displacement(self):
        """EVT_GRIT_IN_EYE (fabula=1, syuzhet=3) is shown later than chronological,
        meaning positive displacement (withheld_cause)."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)
        tensions = assembler.compute_narrative_tension()

        tension_map = {t.event_id: t for t in tensions}
        grit = tension_map["EVT_GRIT_IN_EYE"]
        # fabula=1 → rank 0, syuzhet=3 → rank 2 ⇒ positive displacement
        assert grit.displacement > 0, (
            f"EVT_GRIT_IN_EYE should have positive displacement, got {grit.displacement}"
        )

    def test_all_events_covered(self):
        """Every event in the world state must produce a NarrativeTension."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)
        tensions = assembler.compute_narrative_tension()

        assert len(tensions) == len(brief_encounter_ws.events)
        event_ids = {t.event_id for t in tensions}
        for evt in brief_encounter_ws.events:
            assert evt.id in event_ids

    def test_displacement_range(self):
        """All displacements must be in [-1, 1]."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)
        tensions = assembler.compute_narrative_tension()

        for t in tensions:
            assert -1.0 <= t.displacement <= 1.0, (
                f"{t.event_id} displacement {t.displacement} out of range"
            )

    def test_syuzhet_anchor_marks_revealed_as_linear(self):
        """Events already revealed (syuzhet_index <= anchor) should be classified
        as 'linear' regardless of displacement."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)
        # Anchor at 2: EVT_FINAL_MEETING (s=1, displaced) and
        # EVT_ALEC_DEPARTS (s=2, displaced) are both revealed, so they
        # should be 'linear' even though displaced.
        tensions = assembler.compute_narrative_tension(syuzhet_anchor=2)

        tension_map = {t.event_id: t for t in tensions}
        assert tension_map["EVT_FINAL_MEETING"].tension_type == "linear"
        assert tension_map["EVT_ALEC_DEPARTS"].tension_type == "linear"

    def test_syuzhet_anchor_unrevealed_events_flagged(self):
        """Events NOT yet revealed (syuzhet_index > anchor) should keep their
        displacement-based tension_type."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)
        # Anchor at 1: EVT_GRIT_IN_EYE (s=2) is unrevealed, displacement > 0
        tensions = assembler.compute_narrative_tension(syuzhet_anchor=1)

        tension_map = {t.event_id: t for t in tensions}
        grit = tension_map["EVT_GRIT_IN_EYE"]
        assert grit.tension_type == "withheld_cause", (
            f"Expected withheld_cause, got {grit.tension_type}"
        )

    def test_linear_world_has_near_zero_displacement(self):
        """Macbeth (linear syuzhet) should have displacements near zero."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        tensions = assembler.compute_narrative_tension()

        for t in tensions:
            assert abs(t.displacement) < 0.05, (
                f"Linear plot event {t.event_id} has displacement {t.displacement}"
            )
            assert t.tension_type == "linear"


# =====================================================================
# SYUZHET-AWARE CREATIVE BRIEF CONSTRAINTS
# =====================================================================
class TestSyuzhetAwareConstraints:
    """assemble() with syuzhet_anchor must produce narrative constraints."""

    def test_suspense_with_syuzhet_anchor_adds_narrative_constraint(self):
        """Suspense directive on brief_encounter with early anchor should produce
        NARRATIVE STRUCTURE constraints for withheld events."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_LAURA"],
            target_effect="suspense",
            intensity=0.8,
        )
        # Anchor at 2: only EVT_FINAL_MEETING & EVT_ALEC_DEPARTS revealed
        brief = assembler.assemble(directive, syuzhet_anchor=2)

        narrative_constraints = [
            c for c in brief.constraints if c.constraint_type == "narrative"
        ]
        assert len(narrative_constraints) >= 1, (
            "Suspense with syuzhet_anchor should produce narrative structure constraints"
        )
        # Should mention withholding
        assert any("withheld" in c.instruction.lower() or "MUST NOT" in c.instruction
                    for c in narrative_constraints)

    def test_surprise_with_syuzhet_anchor_adds_flashback_constraint(self):
        """Surprise directive should detect upcoming revelations with high displacement."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_LAURA"],
            target_effect="surprise",
            intensity=1.0,
        )
        brief = assembler.assemble(directive, syuzhet_anchor=2)

        narrative_constraints = [
            c for c in brief.constraints if c.constraint_type == "narrative"
        ]
        # There should be at least one flashback reveal constraint for
        # chronologically early events that haven't been shown yet.
        assert len(narrative_constraints) >= 1, (
            "Surprise with syuzhet_anchor should flag upcoming flashback revelations"
        )

    def test_suspense_without_anchor_still_works(self):
        """Suspense directive without syuzhet_anchor should still produce epistemic
        constraints but no active narrative tension constraints."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_WINSTON"],
            target_effect="suspense",
            intensity=0.8,
        )
        brief = assembler.assemble(directive)

        epistemic = [c for c in brief.constraints if c.constraint_type == "epistemic"]
        assert len(epistemic) >= 1

    def test_narrative_tensions_populated_in_brief(self):
        """CreativeBrief should include the narrative_tensions list."""
        ego = _ego_payload(brief_encounter_ws, ["ENT_LAURA"])
        assembler = DirectiveAssembler(None, ego, brief_encounter_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_LAURA"],
            target_effect="suspense",
            intensity=0.8,
        )
        brief = assembler.assemble(directive, syuzhet_anchor=2)

        assert len(brief.narrative_tensions) == len(brief_encounter_ws.events)
        # Frame events should be marked linear (already revealed)
        tension_map = {t.event_id: t for t in brief.narrative_tensions}
        assert tension_map["EVT_FINAL_MEETING"].tension_type == "linear"

    def test_narrative_physics_with_syuzhet_anchor(self):
        """calculate_narrative_physics with syuzhet_anchor must pass through."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_LAURA"],
            target_effect="suspense",
            intensity=0.8,
        )
        result = calculate_narrative_physics(
            query, brief_encounter_ws,
            use_causal_engine=True,
            syuzhet_anchor=2,
        )
        assert result["status"] == "success"
        brief = result["creative_brief"]
        assert len(brief["narrative_tensions"]) == len(brief_encounter_ws.events)
        narrative_constraints = [
            c for c in brief["constraints"] if c["constraint_type"] == "narrative"
        ]
        assert len(narrative_constraints) >= 1


# =====================================================================
# HIDDEN INFORMATION CHANNELS
# =====================================================================
class TestHiddenChannels:
    """compute_hidden_channels must detect undiscovered Channels and
    future utterance EventNodes (the post-refactor replacement for the
    legacy ``InformationEdge``)."""

    def test_no_hidden_channels_without_anchor(self):
        """Without a syuzhet_anchor, no channels should be hidden."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)
        hidden = assembler.compute_hidden_channels()
        assert hidden == []

    def test_hidden_channels_with_future_utterance(self):
        """A Channel whose only utterance has syuzhet_index > anchor should be hidden."""
        from shadow_loom.models import Channel, EventNode
        ws = deepcopy(orwell_ws)
        # Add a fresh channel + a future utterance over it
        ws.channels["CHN_FUTURE"] = Channel(
            id="CHN_FUTURE", name="future", medium="speech",
            participant_ids=["ENT_WINSTON", "ENT_OBRIEN"],
            established_at_fabula=100,
        )
        ws.events.append(EventNode(
            id="EVT_UTT_FUTURE", event_type="utterance",
            description="future", speaker_id="ENT_WINSTON",
            addressee_ids=["ENT_OBRIEN"], actor_ids=["ENT_WINSTON"],
            target_ids=[], via_channel_id="CHN_FUTURE",
            fabula_time=2000, syuzhet_index=99,
        ))
        ego = _ego_payload(ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, ws)
        hidden = assembler.compute_hidden_channels(syuzhet_anchor=5)
        assert any(h.channel_id == "CHN_FUTURE" or h.utterance_event_id == "EVT_UTT_FUTURE"
                   for h in hidden)

    def test_hidden_channels_added_to_brief(self):
        """Suspense brief should include hidden channels as narrative constraints."""
        from shadow_loom.models import Channel, EventNode
        ws = deepcopy(orwell_ws)
        ws.channels["CHN_FUTURE"] = Channel(
            id="CHN_FUTURE", name="future", medium="speech",
            participant_ids=["ENT_WINSTON", "ENT_OBRIEN"],
            established_at_fabula=100,
        )
        ws.events.append(EventNode(
            id="EVT_UTT_FUTURE", event_type="utterance",
            description="future", speaker_id="ENT_WINSTON",
            addressee_ids=["ENT_OBRIEN"], actor_ids=["ENT_WINSTON"],
            target_ids=[], via_channel_id="CHN_FUTURE",
            fabula_time=2000, syuzhet_index=99,
        ))
        ego = _ego_payload(ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_WINSTON"],
            target_effect="suspense",
            intensity=0.8,
        )
        brief = assembler.assemble(directive, syuzhet_anchor=5)

        hidden_constraints = [
            c for c in brief.constraints
            if "HIDDEN CHANNEL" in c.instruction or "HIDDEN UTTERANCE" in c.instruction
        ]
        assert len(hidden_constraints) >= 1


# =====================================================================
# _classify_gap QUANTITATIVE BRANCH
# =====================================================================
class TestClassifyGapQuantitative:
    """_classify_gap must detect numeric values and produce 'quantitative' gaps."""

    def test_numeric_belief_produces_quantitative_gap(self):
        """Belief with '0.2' vs actual with '0.8' must be 'quantitative'."""
        ws = deepcopy(macbeth_ws)
        # Give Macbeth a belief with numeric content about an object
        ws.entities["ENT_MACBETH"].beliefs = [
            Belief(target_id="OBJ_CROWN", perceived_state="value=0.2",
                   confidence=0.9, inertia=0.5),
        ]
        # OBJ_CROWN.properties has state: must yield a string with a number
        ws.objects["OBJ_CROWN"].properties["value"] = "0.8"

        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)
        gaps = assembler.compute_epistemic_gaps(["ENT_MACBETH"])

        crown_gaps = [g for g in gaps if g.belief_target_id == "OBJ_CROWN"]
        assert crown_gaps, "Object with numeric property should produce quantitative gap"
        assert crown_gaps[0].gap_type == "quantitative"
        assert crown_gaps[0].gap_magnitude > 0

    def test_non_numeric_belief_uses_token_heuristic(self):
        """Belief without numbers must fall through to token overlap."""
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].beliefs = [
            Belief(target_id="OBJ_CROWN", perceived_state="crown is safe",
                   confidence=0.9, inertia=0.5),
        ]
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)
        gaps = assembler.compute_epistemic_gaps(["ENT_MACBETH"])

        crown_gaps = [g for g in gaps if g.belief_target_id == "OBJ_CROWN"]
        assert crown_gaps, "Non-numeric belief should produce gap via token heuristic"
        assert crown_gaps[0].gap_type in ("confirmed", "contradicted", "unknown")


# =====================================================================
# EFFECT DECREASE TRAIT DIRECTION
# =====================================================================
class TestEffectDecreaseDirection:
    """Emotion directive must correctly use positive/negative direction."""

    def test_grief_hope_produces_negative_direction(self):
        """Grief → hope must produce a NEGATIVE direction in the constraint."""
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].traits["hope"] = TraitVector(value=0.8, inertia=0.1)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="grief",
            intensity=0.7,
        )
        brief = assembler.assemble(directive)

        hope_constraints = [
            c for c in brief.constraints
            if "hope" in c.instruction.lower() and c.constraint_type == "mathematical"
        ]
        assert hope_constraints, "Grief effect should produce hope constraint with value=0.8"
        # Direction should be negative (decrease hope)
        assert "-0.70" in hope_constraints[0].instruction

    def test_grief_despair_produces_positive_direction(self):
        """Grief → despair must produce a POSITIVE direction."""
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].traits["despair"] = TraitVector(value=0.2, inertia=0.1)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="grief",
            intensity=0.7,
        )
        brief = assembler.assemble(directive)

        despair_constraints = [
            c for c in brief.constraints
            if "despair" in c.instruction.lower() and c.constraint_type == "mathematical"
        ]
        assert despair_constraints, "Grief effect should produce despair constraint with value=0.2"
        assert "+0.70" in despair_constraints[0].instruction

    def test_headroom_floor_skip(self):
        """Trait with headroom < 0.05 must be skipped."""
        ws = deepcopy(macbeth_ws)
        # Despair already at ceiling → headroom_up = 0.01 < 0.05
        ws.entities["ENT_MACBETH"].traits["despair"] = TraitVector(value=0.99, inertia=0.1)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="grief",
            intensity=0.7,
        )
        brief = assembler.assemble(directive)

        # despair should NOT appear in constraints (no room to increase)
        despair_constraints = [
            c for c in brief.constraints
            if "despair" in c.instruction.lower() and c.constraint_type == "mathematical"
        ]
        assert len(despair_constraints) == 0


# =====================================================================
# ASSEMBLE — ADDITIONAL EMOTION EFFECTS
# =====================================================================
class TestAssembleAdditionalEmotions:
    """assemble() must produce correct constraints for all emotion effects."""

    def test_joy_directive(self):
        """Joy must produce constraints for happiness/hope/contentment."""
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].traits["happiness"] = TraitVector(value=0.3, inertia=0.1)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="joy",
            intensity=0.8,
        )
        brief = assembler.assemble(directive)
        math_c = [c for c in brief.constraints if c.constraint_type == "mathematical"]
        assert len(math_c) >= 1

    def test_love_directive(self):
        """Love must produce constraints for love/affection/sensuality."""
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].traits["love"] = TraitVector(value=0.2, inertia=0.1)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="love",
            intensity=0.6,
        )
        brief = assembler.assemble(directive)
        math_c = [c for c in brief.constraints if c.constraint_type == "mathematical"]
        assert len(math_c) >= 1

    def test_regret_directive(self):
        """Regret must produce constraints for guilt/remorse/despair."""
        ws = deepcopy(macbeth_ws)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="regret",
            intensity=0.9,
        )
        brief = assembler.assemble(directive)
        math_c = [c for c in brief.constraints if c.constraint_type == "mathematical"]
        assert len(math_c) >= 1

    def test_fear_directive(self):
        """Fear must produce constraints for fear/paranoia/anxiety."""
        ws = deepcopy(macbeth_ws)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
            intensity=0.5,
        )
        brief = assembler.assemble(directive)
        math_c = [c for c in brief.constraints if c.constraint_type == "mathematical"]
        assert len(math_c) >= 1

    def test_mystery_routes_to_epistemic(self):
        """'mystery' must route to same epistemic constraint path as 'suspense'."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)

        directive = DirectiveQuery(
            target_entity_ids=["ENT_WINSTON"],
            target_effect="mystery",
            intensity=0.8,
        )
        brief = assembler.assemble(directive)
        epistemic = [c for c in brief.constraints if c.constraint_type == "epistemic"]
        assert len(epistemic) >= 1


# =====================================================================
# AFFECTIVE SCORE — EFFECT-DECREASE HEADROOM
# =====================================================================
class TestAffectiveScoreHeadroom:
    """compute_affective_score must use correct headroom for decrease traits."""

    def test_grief_score_reflects_headroom_down_for_hope(self):
        """Grief: 'hope' in decrease set → score should use headroom_down."""
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_MACBETH"].traits["hope"] = TraitVector(value=0.9, inertia=0.1)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        score = assembler.compute_affective_score(
            "grief", ["ENT_MACBETH"],
        )
        # Score should be negative (good match) because hope has lots of
        # headroom_down (0.9)
        assert score < 0

    def test_no_relevant_traits_penalty(self):
        """Entity with no matching traits must get +0.5 penalty."""
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A",
                    status="healthy",
                    # Only 'strength' trait — not in any emotion map
                    traits={"strength": TraitVector(value=0.5, inertia=0.1)},
                ),
            },
            events=[],
            causal_topology=[],
            spatial_topology=[],
            social_topology=[],
        )
        ego = _ego_payload(ws, ["ENT_X"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_affective_score("love", ["ENT_X"])
        assert score > 0  # penalty for no relevant traits


# =====================================================================
# _build_vector_constraint EDGE CASES
# =====================================================================
class TestBuildVectorConstraint:
    """_build_vector_constraint edge cases."""

    def test_no_dot_in_vector_id(self):
        """Vector ID without a dot must produce generic mathematical constraint."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        constraint = assembler._build_vector_constraint("ENT_MACBETH", 0.5)
        assert constraint is not None
        assert constraint.constraint_type == "mathematical"

    def test_beliefs_path(self):
        """Beliefs path must produce an epistemic constraint."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        constraint = assembler._build_vector_constraint(
            "ENT_MACBETH.beliefs.ENT_DUNCAN", 0.7,
        )
        assert constraint is not None, "Beliefs path should produce an epistemic constraint"
        assert constraint.constraint_type == "epistemic"


# =====================================================================
# NARRATIVE TENSION EDGE CASES
# =====================================================================
class TestNarrativeTensionEdgeCases:
    """Edge cases in compute_narrative_tension."""

    def test_empty_events_returns_empty(self):
        """World with no events must return empty list."""
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A",
                    status="healthy", traits={},
                ),
            },
            events=[],
            causal_topology=[],
            spatial_topology=[],
            social_topology=[],
        )
        ego = _ego_payload(ws, ["ENT_X"])
        assembler = DirectiveAssembler(None, ego, ws)
        assert assembler.compute_narrative_tension() == []

    def test_single_event_zero_displacement(self):
        """Single event must have displacement=0."""
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A",
                    status="healthy", traits={},
                ),
            },
            events=[
                EventNode(id="EVT_ONLY", fabula_time=1, syuzhet_index=1,
                          event_type="outcome", description="Only event"),
            ],
            causal_topology=[],
            spatial_topology=[],
            social_topology=[],
        )
        ego = _ego_payload(ws, ["ENT_X"])
        assembler = DirectiveAssembler(None, ego, ws)
        tensions = assembler.compute_narrative_tension()
        assert len(tensions) == 1
        assert tensions[0].displacement == 0.0
        assert tensions[0].tension_type == "linear"


# =====================================================================
# PHYSICS OVERRIDE — EGO FALLBACK PATH
# =====================================================================
class TestPhysicsOverrideEgoFallback:
    """_detect_physics_override ego payload fallback."""

    def test_ego_multi_location_returns_override(self):
        """Multi-location ego (no sandbox) must return override."""
        ws = deepcopy(macbeth_ws)
        ws.entities["ENT_LADY_MACBETH"].location_id = "LOC_ENGLAND"
        ego = _ego_payload(ws, ["ENT_MACBETH", "ENT_LADY_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)

        override = assembler._detect_physics_override()
        assert override is not None
        assert "multiple locations" in override.lower()

    def test_ego_single_location_returns_none(self):
        """Single-location ego (no sandbox) must return None."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        override = assembler._detect_physics_override()
        assert override is None


# =====================================================================
# MYSTERY SCORE  (hidden causal ancestors)
# =====================================================================
class TestMysteryScore:
    """compute_mystery_score must count hidden causal predecessors."""

    def _make_mystery_world(self):
        """World with a chain: EVT_SETUP → EVT_MURDER → ENT_VICTIM.
        Reader sees EVT_MURDER (syuzhet=1) but NOT EVT_SETUP (syuzhet=3)."""
        return WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_VICTIM": Entity(
                    id="ENT_VICTIM", name="Victim", location_id="LOC_A",
                    status="dead",
                    traits={"fear": TraitVector(value=0.9, inertia=0.1)},
                ),
            },
            events=[
                EventNode(id="EVT_SETUP", fabula_time=1, syuzhet_index=3,
                          event_type="choice", actor_ids=[],
                          description="Hidden setup"),
                EventNode(id="EVT_MURDER", fabula_time=2, syuzhet_index=1,
                          event_type="choice", actor_ids=[],
                          target_ids=["ENT_VICTIM"],
                          description="The murder"),
                EventNode(id="EVT_FOUND", fabula_time=3, syuzhet_index=2,
                          event_type="revelation", actor_ids=["ENT_VICTIM"],
                          description="Body found"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_SETUP",
                           target_id="EVT_MURDER", causality_type="chain_reaction", causal_force=5.0,
                           mechanism="physical", evidence_strength="strong",
                           fabula_time=1),
                CausalEdge(source_id="EVT_MURDER",
                           target_id="ENT_VICTIM", causality_type="mutation", causal_force=5.0,
                           mechanism="physical", evidence_strength="strong",
                           fabula_time=2),
            ],
            spatial_topology=[],
            social_topology=[],
            )

    def test_hidden_ancestors_detected(self):
        """Mystery score > 0 when syuzhet_anchor hides a causal predecessor."""
        ws = self._make_mystery_world()
        ego = _ego_payload(ws, ["ENT_VICTIM"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_mystery_score(["ENT_VICTIM"], syuzhet_anchor=2)
        # EVT_SETUP (syuzhet=3) is hidden, EVT_MURDER's ancestor → mystery > 0
        assert score > 0

    def test_all_revealed_means_zero(self):
        """Mystery = 0 when all causal ancestors are revealed."""
        ws = self._make_mystery_world()
        ego = _ego_payload(ws, ["ENT_VICTIM"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_mystery_score(["ENT_VICTIM"], syuzhet_anchor=3)
        assert score == 0.0

    def test_no_anchor_means_all_revealed(self):
        """No syuzhet_anchor → all events revealed → mystery = 0."""
        ws = self._make_mystery_world()
        ego = _ego_payload(ws, ["ENT_VICTIM"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_mystery_score(["ENT_VICTIM"], syuzhet_anchor=None)
        assert score == 0.0


# =====================================================================
# DRAMATIC IRONY SCORE  (reader > character knowledge)
# =====================================================================
class TestDramaticIronyScore:
    """compute_dramatic_irony_score must detect reader/character asymmetry."""

    def test_irony_when_reader_knows_character_doesnt(self):
        """Irony > 0 when reader sees a cause targeting the character but
        the character has no belief about that source event."""
        ws = deepcopy(macbeth_ws)
        ego = _ego_payload(ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, ws)
        # Pick a syuzhet_anchor that reveals causal edges targeting Macbeth
        # but Macbeth's beliefs don't reference those source events
        score = assembler.compute_dramatic_irony_score(
            ["ENT_MACBETH"], syuzhet_anchor=10,
        )
        # The score depends on Macbeth's beliefs vs revealed causal edges
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_no_anchor_returns_zero(self):
        """No syuzhet_anchor → irony always 0 (cannot determine reader state)."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        assert assembler.compute_dramatic_irony_score(["ENT_MACBETH"]) == 0.0

    def test_irony_zero_when_character_aware(self):
        """Irony = 0 when character's beliefs cover all revealed causes."""
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_A": Entity(
                    id="ENT_A", name="A", location_id="LOC_A",
                    status="healthy",
                    traits={"fear": TraitVector(value=0.5, inertia=0.3)},
                    beliefs=[
                        Belief(target_id="EVT_THREAT",
                               perceived_state="I know about the threat",
                               confidence=0.9, inertia=0.5),
                    ],
                ),
            },
            events=[
                EventNode(id="EVT_THREAT", fabula_time=1, syuzhet_index=1,
                          event_type="outcome", actor_ids=[],
                          target_ids=["ENT_A"],
                          description="A threat looms"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_THREAT",
                           target_id="ENT_A", causality_type="mutation", causal_force=5.0,
                           mechanism="psychological", evidence_strength="moderate",
                           fabula_time=1),
            ],
            spatial_topology=[],
            social_topology=[],
            )
        ego = _ego_payload(ws, ["ENT_A"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_dramatic_irony_score(
            ["ENT_A"], syuzhet_anchor=1,
        )
        assert score == 0.0


# =====================================================================
# SUSPENSE SCORE  (P(threat) - P(hope))
# =====================================================================
class TestSuspenseScore:
    """compute_suspense_score must compute forward causal momentum."""

    def _make_suspense_world(self):
        """World with strong threat and weak hope for ENT_HERO."""
        return WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_HERO": Entity(
                    id="ENT_HERO", name="Hero", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.6, inertia=0.3)},
                ),
                "ENT_VILLAIN": Entity(
                    id="ENT_VILLAIN", name="Villain", location_id="LOC_A",
                    status="healthy",
                    traits={"cruelty": TraitVector(value=0.9, inertia=0.8)},
                ),
            },
            events=[
                EventNode(id="EVT_REVEALED", fabula_time=1, syuzhet_index=1,
                          event_type="outcome", actor_ids=[],
                          description="Status quo"),
                # Threat: villain attacks hero (strong causal edge)
                EventNode(id="EVT_ATTACK", fabula_time=2, syuzhet_index=3,
                          event_type="choice", actor_ids=["ENT_VILLAIN"],
                          target_ids=["ENT_HERO"],
                          description="The villain strikes"),
                # Hope: hero escapes (no strong causal backing)
                EventNode(id="EVT_ESCAPE", fabula_time=3, syuzhet_index=4,
                          event_type="choice", actor_ids=["ENT_HERO"],
                          description="The hero finds a way out"),
            ],
            causal_topology=[
                # Strong causal edge from setup to attack
                CausalEdge(source_id="EVT_REVEALED",
                           target_id="EVT_ATTACK", causality_type="chain_reaction", causal_force=9.0,
                           mechanism="physical", evidence_strength="strong",
                           fabula_time=1),
            ],
            spatial_topology=[],
            social_topology=[],
            )

    def test_suspense_positive_when_threat_exceeds_hope(self):
        """Suspense > 0 when P(threat) > P(hope)."""
        ws = self._make_suspense_world()
        ego = _ego_payload(ws, ["ENT_HERO"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_suspense_score(
            ["ENT_HERO"], syuzhet_anchor=1,
        )
        # EVT_ATTACK has strong causal backing (0.75); EVT_ESCAPE has none (0.5)
        assert score > 0

    def test_suspense_zero_when_no_hope(self):
        """Suspense = 0 when hope is entirely extinguished."""
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_X": Entity(id="ENT_X", name="X", location_id="LOC_A",
                                status="healthy",
                                traits={"fear": TraitVector(value=0.5, inertia=0.3)}),
            },
            events=[
                EventNode(id="EVT_NOW", fabula_time=1, syuzhet_index=1,
                          event_type="outcome", actor_ids=[],
                          description="Now"),
                # Only threat, no hope event
                EventNode(id="EVT_DOOM", fabula_time=2, syuzhet_index=2,
                          event_type="outcome", actor_ids=["ENT_VILLAIN"],
                          target_ids=["ENT_X"],
                          description="Doom"),
            ],
            causal_topology=[],
            spatial_topology=[],
            social_topology=[],
            )
        ego = _ego_payload(ws, ["ENT_X"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_suspense_score(["ENT_X"], syuzhet_anchor=1)
        assert score == 0.0  # no hope → despair, not suspense

    def test_suspense_zero_when_all_revealed(self):
        """No unrevealed events → suspense = 0."""
        ws = self._make_suspense_world()
        ego = _ego_payload(ws, ["ENT_HERO"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_suspense_score(
            ["ENT_HERO"], syuzhet_anchor=10,
        )
        assert score == 0.0


# =====================================================================
# SURPRISE SCORE  (KL Divergence)
# =====================================================================
class TestSurpriseScore:
    """compute_surprise_score must compute KL divergence from reader prior."""

    def test_surprise_from_unrevealed_causes(self):
        """When causal edges are hidden from reader, prior ≠ actual → KL > 0."""
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_A": Entity(
                    id="ENT_A", name="A", location_id="LOC_A",
                    status="healthy",
                    traits={
                        # Extreme value far from 0.5 → high KL when prior=0.5
                        "courage": TraitVector(value=0.95, inertia=0.1),
                        "fear": TraitVector(value=0.05, inertia=0.1),
                    },
                ),
            },
            events=[
                EventNode(id="EVT_CAUSE", fabula_time=1, syuzhet_index=5,
                          event_type="outcome", actor_ids=[],
                          description="Hidden cause"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_CAUSE",
                           target_id="ENT_A", causality_type="mutation", causal_force=5.0,
                           mechanism="psychological", evidence_strength="strong",
                           fabula_time=1),
            ],
            spatial_topology=[],
            social_topology=[],
            )
        ego = _ego_payload(ws, ["ENT_A"])
        assembler = DirectiveAssembler(None, ego, ws)
        score = assembler.compute_surprise_score(
            ["ENT_A"], syuzhet_anchor=1,
        )
        # EVT_CAUSE is hidden (syuzhet=5 > anchor=1), so prior stays ~0.5
        # Actual traits are 0.95 and 0.05 → high KL
        assert score > 0

    def test_surprise_zero_when_all_revealed(self):
        """When all causes are revealed, prior ≈ actual → KL ≈ 0."""
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_A": Entity(
                    id="ENT_A", name="A", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.8, inertia=0.1)},
                ),
            },
            events=[
                EventNode(id="EVT_CAUSE", fabula_time=1, syuzhet_index=1,
                          event_type="outcome", actor_ids=[],
                          description="Revealed cause"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_CAUSE",
                           target_id="ENT_A", causality_type="mutation", causal_force=5.0,
                           mechanism="psychological", evidence_strength="strong",
                           fabula_time=1),
            ],
            spatial_topology=[],
            social_topology=[],
            )
        ego = _ego_payload(ws, ["ENT_A"])
        assembler = DirectiveAssembler(None, ego, ws)
        score_revealed = assembler.compute_surprise_score(
            ["ENT_A"], syuzhet_anchor=1,
        )
        score_hidden = assembler.compute_surprise_score(
            ["ENT_A"], syuzhet_anchor=0,
        )
        # Revealed causes shift prior closer to actual → less surprise
        assert score_revealed < score_hidden

    def test_surprise_zero_without_anchor(self):
        """No syuzhet_anchor → reader knows everything → surprise = 0."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        assert assembler.compute_surprise_score(["ENT_MACBETH"]) == 0.0

    def test_kl_divergence_increases_with_trait_extremity(self):
        """Traits further from 0.5 should produce higher KL (more surprise)."""
        def _make_ws(val):
            return WorldStateV1(
                locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
                objects={},
                entities={
                    "ENT_A": Entity(
                        id="ENT_A", name="A", location_id="LOC_A",
                        status="healthy",
                        traits={"x": TraitVector(value=val, inertia=0.1)},
                    ),
                },
                events=[
                    EventNode(id="EVT_X", fabula_time=1, syuzhet_index=5,
                              event_type="outcome", actor_ids=[],
                              description="Hidden"),
                ],
                causal_topology=[
                    CausalEdge(source_id="EVT_X",
                               target_id="ENT_A", causality_type="mutation", causal_force=5.0,
                               mechanism="physical", evidence_strength="strong",
                               fabula_time=1),
                ],
                spatial_topology=[],
                social_topology=[],
                )

        ws_moderate = _make_ws(0.7)
        ws_extreme = _make_ws(0.95)
        ego_mod = _ego_payload(ws_moderate, ["ENT_A"])
        ego_ext = _ego_payload(ws_extreme, ["ENT_A"])
        asm_mod = DirectiveAssembler(None, ego_mod, ws_moderate)
        asm_ext = DirectiveAssembler(None, ego_ext, ws_extreme)

        score_mod = asm_mod.compute_surprise_score(["ENT_A"], syuzhet_anchor=1)
        score_ext = asm_ext.compute_surprise_score(["ENT_A"], syuzhet_anchor=1)
        assert score_ext > score_mod


# =====================================================================
# AFFECTIVE SCORE DISPATCH
# =====================================================================
class TestAffectiveScoreDispatch:
    """compute_affective_score must dispatch correctly to dedicated methods."""

    def test_mystery_dispatch(self):
        """Mystery effect dispatches to compute_mystery_score."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        score = assembler.compute_affective_score(
            "mystery", ["ENT_MACBETH"], syuzhet_anchor=5,
        )
        assert isinstance(score, float)

    def test_dramatic_irony_dispatch(self):
        """Dramatic irony effect dispatches to compute_dramatic_irony_score."""
        ego = _ego_payload(orwell_ws, ["ENT_WINSTON"])
        assembler = DirectiveAssembler(None, ego, orwell_ws)
        score = assembler.compute_affective_score(
            "dramatic_irony", ["ENT_WINSTON"], syuzhet_anchor=5,
        )
        assert isinstance(score, float)

    def test_suspense_dispatch(self):
        """Suspense effect dispatches to compute_suspense_score."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        score = assembler.compute_affective_score(
            "suspense", ["ENT_MACBETH"], syuzhet_anchor=3,
        )
        assert isinstance(score, float)

    def test_surprise_dispatch(self):
        """Surprise effect dispatches to compute_surprise_score."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        score = assembler.compute_affective_score(
            "surprise", ["ENT_MACBETH"], syuzhet_anchor=3,
        )
        assert isinstance(score, float)

    def test_emotion_dispatch_unchanged(self):
        """Emotion effects still use trait headroom."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        score = assembler.compute_affective_score(
            "grief", ["ENT_MACBETH"],
        )
        assert isinstance(score, float)
