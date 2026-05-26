# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-8 audit \u2014 unchecked Do-handler paths.

Five defects identified during the round-8 audit of the remaining
typed do-target handlers \u2014 the ones that do NOT touch the create /
sever / mutate-state surfaces already covered by rounds 6/7:

* **R8-F1** \u2014 ``_apply_do_proposition`` silently recorded a
  ``PropositionMutation`` even when the proposition was missing from
  the canonical catalogue. Pollutes the audit log and the sandbox's
  ``proposition_clamps`` with phantom truth pins. Now WARN + return.
* **R8-F2** \u2014 ``_apply_do_world_trait`` mutated only the sandbox
  node, never mirroring to
  ``world_state.world_traits[wt_id].magnitude``. Asymmetric vs every
  other handler; the canonical world stayed stale.
* **R8-F3** \u2014 ``_apply_do_concern`` mutated only the sandbox; the
  next re-extraction read the canonical
  ``Entity.concerns`` and silently restored the pre-surgery state.
* **R8-F4** \u2014 ``_apply_do_belief`` had the same missing-mirror
  defect. Engine-side surgery was authoritative for the current
  propagation step but did not survive a merge.
* **R8-F5** \u2014 ``_apply_do_channel`` silently no-op'd the canonical
  mirror when the channel was missing from ``world.channels`` (no
  WARNING \u2014 every other handler warns on a missed mirror).
"""
from __future__ import annotations

import logging

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import (
    Belief,
    Channel,
    Concern,
    Entity,
    EventNode,
    GlobalTrait,
    Location,
    Proposition,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.query_models import (
    DoBelief,
    DoChannel,
    DoConcern,
    DoProposition,
    DoWorldTrait,
)


def _make_world() -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.2)},
                beliefs=[],
                concerns=[],
            ),
        },
        events=[
            EventNode(
                id="EVT_ANCHOR", fabula_time=10, syuzhet_index=10,
                event_type="outcome", actor_ids=["ENT_ALICE"], target_ids=[],
                description="Anchor event.",
            ),
        ],
        causal_topology=[],
        spatial_topology=[],
        social_topology=[],
        channels={},
        propositions=[],
        world_traits={
            "WORLD_FOG": GlobalTrait(
                id="WORLD_FOG", name="fog of war",
                description="ambient",
                category="environmental",
                magnitude=TraitVector(value=0.3, inertia=0.4),
                affected_domains=["physical"],
            ),
        },
    )


def _engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    return CausalPhysicsEngine(sandbox, ws)


# ---------------------------------------------------------------------
# R8-F1 \u2014 DoProposition refuses missing proposition
# ---------------------------------------------------------------------
class TestDoPropositionMissingTarget:
    def test_missing_proposition_does_not_record_mutation(self, caplog):
        ws = _make_world()
        eng = _engine(ws)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoProposition(
                proposition_id="PROP_GHOST", truth=True, fabula_time=10,
                propagate_to_beliefs=False,
            )])
        # No phantom mutation row.
        assert eng._proposition_mutations == [], (
            "DoProposition on a missing proposition must not emit a mutation row"
        )
        # No phantom clamp on the sandbox graph.
        assert not eng.sandbox.graph.get("proposition_clamps"), (
            "DoProposition on a missing proposition must not pollute "
            "proposition_clamps"
        )
        assert any(
            "not in world_state.propositions" in r.getMessage()
            for r in caplog.records
        )

    def test_existing_proposition_still_clamps(self):
        ws = _make_world()
        ws.propositions = [Proposition(
            world_id="factual",
            proposition_id="PROP_REAL",
            kind="outcome",
            referent_ids=[],
            description="Alice escapes",
            audience_default_prior=0.5,
            stakes=0.5,
            truth_at_fabula={5: False},
        )]
        eng = _engine(ws)
        eng.apply_do_targets([DoProposition(
            proposition_id="PROP_REAL", truth=True, fabula_time=10,
            propagate_to_beliefs=False,
        )])
        prop = next(p for p in ws.propositions if p.proposition_id == "PROP_REAL")
        assert prop.truth_at_fabula.get(10) is True
        assert len(eng._proposition_mutations) == 1


# ---------------------------------------------------------------------
# R8-F2 \u2014 DoWorldTrait mirrors to canonical
# ---------------------------------------------------------------------
class TestDoWorldTraitCanonicalMirror:
    def test_value_mirrored(self):
        ws = _make_world()
        eng = _engine(ws)
        eng.apply_do_targets([DoWorldTrait(
            world_trait_id="WORLD_FOG", value=0.9,
        )])
        assert ws.world_traits["WORLD_FOG"].magnitude.value == 0.9, (
            "canonical world.world_traits must reflect the surgery"
        )

    def test_domains_mirrored(self):
        ws = _make_world()
        eng = _engine(ws)
        eng.apply_do_targets([DoWorldTrait(
            world_trait_id="WORLD_FOG", value=0.5,
            affected_domains_add=["psychological"],
            affected_domains_remove=["physical"],
        )])
        domains = ws.world_traits["WORLD_FOG"].affected_domains
        assert "psychological" in domains
        assert "physical" not in domains


# ---------------------------------------------------------------------
# R8-F3 \u2014 DoConcern mirrors to canonical entity
# ---------------------------------------------------------------------
class TestDoConcernCanonicalMirror:
    def test_salience_mirrored(self):
        ws = _make_world()
        ws.propositions = [Proposition(
            world_id="factual",
            proposition_id="PROP_X",
            kind="outcome", referent_ids=[],
            description="anchor", audience_default_prior=0.5, stakes=0.5,
            truth_at_fabula={},
        )]
        ws.entities["ENT_ALICE"].concerns = [Concern(
            world_id="factual",
            concern_id="CCN_FEAR_DEATH",
            proposition_id="PROP_X",
            polarity="fear", salience=0.4,
        )]
        eng = _engine(ws)
        eng.apply_do_targets([DoConcern(
            holder_id="ENT_ALICE",
            concern_id="CCN_FEAR_DEATH",
            salience=0.95,
        )])
        cc = ws.entities["ENT_ALICE"].concerns[0]
        assert cc.salience == 0.95, (
            "canonical entity concerns must reflect the surgery"
        )

    def test_polarity_mirrored(self):
        ws = _make_world()
        ws.propositions = [Proposition(
            world_id="factual",
            proposition_id="PROP_X",
            kind="outcome", referent_ids=[],
            description="anchor", audience_default_prior=0.5, stakes=0.5,
            truth_at_fabula={},
        )]
        ws.entities["ENT_ALICE"].concerns = [Concern(
            world_id="factual",
            concern_id="CCN_X",
            proposition_id="PROP_X",
            polarity="fear", salience=0.5,
        )]
        eng = _engine(ws)
        eng.apply_do_targets([DoConcern(
            holder_id="ENT_ALICE",
            concern_id="CCN_X",
            polarity="desire",
        )])
        assert ws.entities["ENT_ALICE"].concerns[0].polarity == "desire"


# ---------------------------------------------------------------------
# R8-F4 \u2014 DoBelief mirrors to canonical entity
# ---------------------------------------------------------------------
class TestDoBeliefCanonicalMirror:
    def test_existing_belief_mirrored(self):
        ws = _make_world()
        ws.entities["ENT_ALICE"].beliefs = [Belief(
            target_id="ENT_BOB",
            perceived_state="Bob is dangerous",
            confidence=0.5,
            inertia=0.3,
            evidence_strength="moderate",
        )]
        eng = _engine(ws)
        eng.apply_do_targets([DoBelief(
            holder_id="ENT_ALICE",
            target_id="ENT_BOB",
            confidence=0.95,
        )])
        cb = ws.entities["ENT_ALICE"].beliefs[0]
        assert cb.confidence == 0.95, (
            "canonical entity beliefs must reflect the surgery"
        )
        assert cb.acquired_via_event_id == "DO_OPERATOR"

    def test_new_belief_appended_to_canonical(self):
        ws = _make_world()
        assert ws.entities["ENT_ALICE"].beliefs == []
        eng = _engine(ws)
        eng.apply_do_targets([DoBelief(
            holder_id="ENT_ALICE",
            target_id="ENT_GHOST",
            perceived_state="There is a ghost",
            confidence=0.8,
        )])
        beliefs = ws.entities["ENT_ALICE"].beliefs
        assert len(beliefs) == 1
        assert beliefs[0].target_id == "ENT_GHOST"
        assert beliefs[0].confidence == 0.8


# ---------------------------------------------------------------------
# R8-F5 \u2014 DoChannel warns when canonical channel missing
# ---------------------------------------------------------------------
class TestDoChannelCanonicalWarning:
    def test_missing_canonical_channel_warns(self, caplog):
        ws = _make_world()
        # Add the channel to the sandbox manually so the sandbox-side
        # mutation can run; the canonical world remains empty so the
        # mirror is what is being tested.
        eng = _engine(ws)
        eng.sandbox.add_node(
            "CHN_PHANTOM", node_type="Channel",
            terminated_at_fabula=None, intelligibility={},
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoChannel(
                channel_id="CHN_PHANTOM", active=False, fabula_time=10,
            )])
        assert any(
            "missing from world.channels" in r.getMessage()
            and "CHN_PHANTOM" in r.getMessage()
            for r in caplog.records
        )

    def test_canonical_channel_present_no_warning(self, caplog):
        ws = _make_world()
        ws.channels["CHN_REAL"] = Channel(
            id="CHN_REAL", name="line", medium="telephone",
            participant_ids=["ENT_ALICE"], directionality="duplex",
            intelligibility={},
        )
        eng = _engine(ws)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoChannel(
                channel_id="CHN_REAL", active=False, fabula_time=10,
            )])
        assert not any(
            "missing from world.channels" in r.getMessage()
            for r in caplog.records
        )
        assert ws.channels["CHN_REAL"].terminated_at_fabula == 10
