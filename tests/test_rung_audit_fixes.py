# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the three Pearl-rung audit fixes:

1. **Branch-safe ``_apply_deletions``** — a shadow-branch merge whose
   ``removed_*`` fields name factual-tagged elements must NOT delete
   them, only emitting an INFO ``[merge·delete]`` skip log instead.
2. **Multi-ego POV** — ``_resolve_pov_policy`` collapses single-target
   queries to ``"single"`` policy and lifts multi-target queries to
   ``"rotating"``, filtering out the ``ENT_AUDIENCE`` sentinel.
3. **Semantic white-elephant detection** — the auditor's
   ``_paraphrase_match`` flags trigram-overlap rewordings of withheld
   / pruned utterances even when the verbatim substring path misses.
"""
from __future__ import annotations

import logging

import pytest

from shadow_loom.models import (
    Entity,
    EventNode,
    Location,
    TraitVector,
    WorldStateV1,
)


# =====================================================================
# 1. Branch-safe deletions
# =====================================================================
class TestBranchSafeDeletions:
    """A shadow merge must never delete factual-tagged objects, even
    when its ``ChunkTopology`` carries those ids in ``removed_*``."""

    def _world_with_factual_entity(self) -> WorldStateV1:
        return WorldStateV1(
            locations={
                "LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={},
                    world_id="factual",
                ),
            },
            objects={},
            entities={
                "ENT_FACTUAL": Entity(
                    id="ENT_FACTUAL",
                    name="Factual Alice",
                    location_id="LOC_A",
                    status="healthy",
                    traits={"trust": TraitVector(value=0.5, inertia=0.2)},
                    world_id="factual",
                ),
                "ENT_SHADOW": Entity(
                    id="ENT_SHADOW",
                    name="Shadow Bob",
                    location_id="LOC_A",
                    status="healthy",
                    traits={"trust": TraitVector(value=0.5, inertia=0.2)},
                    world_id="shadow",
                ),
            },
            events=[],
            causal_topology=[],
        )

    def test_shadow_merge_does_not_delete_factual_entity(self, caplog):
        from shadow_loom.extract_graph import (
            MergeChangeset,
            _apply_deletions,
        )
        from shadow_loom.ingestion import ChunkTopology

        ws = self._world_with_factual_entity()
        # Shadow merge requesting deletion of a factual-tagged entity.
        topology = ChunkTopology(
            chunk_id="CHK_TEST",
            removed_entity_ids=["ENT_FACTUAL", "ENT_SHADOW"],
        )
        changeset = MergeChangeset()
        with caplog.at_level(logging.INFO):
            _apply_deletions(
                ws, topology,
                changeset=changeset, merge_world_id="shadow",
            )
        # Factual entity preserved; shadow entity removed.
        assert "ENT_FACTUAL" in ws.entities
        assert "ENT_SHADOW" not in ws.entities
        assert changeset.entities_removed == 1
        # INFO log emitted for the skipped cross-branch delete.
        skip_logs = [
            r for r in caplog.records
            if "[merge·delete]" in r.getMessage()
            and "ENT_FACTUAL" in r.getMessage()
        ]
        assert skip_logs, "expected an INFO skip log for ENT_FACTUAL"

    def test_factual_merge_does_not_delete_shadow_entity(self):
        from shadow_loom.extract_graph import (
            MergeChangeset,
            _apply_deletions,
        )
        from shadow_loom.ingestion import ChunkTopology

        ws = self._world_with_factual_entity()
        topology = ChunkTopology(
            chunk_id="CHK_TEST",
            removed_entity_ids=["ENT_SHADOW"],
        )
        changeset = MergeChangeset()
        _apply_deletions(
            ws, topology,
            changeset=changeset, merge_world_id="factual",
        )
        # Shadow entity preserved by the factual merge.
        assert "ENT_SHADOW" in ws.entities
        assert changeset.entities_removed == 0


# =====================================================================
# 2. Multi-ego POV resolution
# =====================================================================
class TestResolvePOVPolicy:
    """``_resolve_pov_policy`` is the single source of truth for the
    pov_lock / additional_pov_locks / pov_policy triad."""

    def test_empty_target_list(self):
        from shadow_loom.generation import _resolve_pov_policy

        pov, extras, policy = _resolve_pov_policy([])
        assert pov is None
        assert extras == []
        assert policy == "single"

    def test_single_target_collapses_to_single(self):
        from shadow_loom.generation import _resolve_pov_policy

        pov, extras, policy = _resolve_pov_policy(["ENT_ALICE"])
        assert pov == "ENT_ALICE"
        assert extras == []
        assert policy == "single"

    def test_multi_target_lifts_to_rotating(self):
        from shadow_loom.generation import _resolve_pov_policy

        pov, extras, policy = _resolve_pov_policy(
            ["ENT_ALICE", "ENT_BOB", "ENT_CARL"],
        )
        assert pov == "ENT_ALICE"
        assert extras == ["ENT_BOB", "ENT_CARL"]
        assert policy == "rotating"

    def test_audience_sentinel_filtered_out(self):
        from shadow_loom.generation import _resolve_pov_policy

        # ENT_AUDIENCE is the reader proxy, not a renderable POV.
        pov, extras, policy = _resolve_pov_policy(
            ["ENT_AUDIENCE", "ENT_ALICE"],
        )
        assert pov == "ENT_ALICE"
        assert extras == []
        assert policy == "single"

    def test_explicit_policy_override_preserved(self):
        from shadow_loom.generation import _resolve_pov_policy

        pov, extras, policy = _resolve_pov_policy(
            ["ENT_ALICE", "ENT_BOB"], explicit_policy="ensemble",
        )
        assert pov == "ENT_ALICE"
        assert extras == ["ENT_BOB"]
        assert policy == "ensemble"


# =====================================================================
# 3. Semantic white-elephant (paraphrase) detection
# =====================================================================
class TestParaphraseLeakDetection:
    """``_paraphrase_match`` must flag trigram-Jaccard rewordings of
    a withheld / pruned utterance even when the verbatim substring
    path misses, while staying silent on unrelated prose."""

    def test_paraphrase_match_detects_high_overlap_rewording(self):
        from shadow_loom.auditor import _paraphrase_match

        canonical = (
            "I poisoned the chalice that Macbeth drank from last night"
        )
        # A reworded version sharing most content words.
        prose_sentences = [
            "Yesterday she poisoned the chalice that Macbeth drank "
            "from last night.",
        ]
        match = _paraphrase_match(canonical, prose_sentences)
        assert match is not None
        sent, score = match
        assert score >= 0.45
        assert "poisoned" in sent.lower()

    def test_paraphrase_match_silent_on_unrelated_prose(self):
        from shadow_loom.auditor import _paraphrase_match

        canonical = (
            "I poisoned the chalice that Macbeth drank from last night"
        )
        prose_sentences = [
            "The morning light filtered through the leaded windows of "
            "the great hall.",
            "Outside, the wind was rising over the moor.",
        ]
        assert _paraphrase_match(canonical, prose_sentences) is None

    def test_paraphrase_match_short_canonical_returns_none(self):
        from shadow_loom.auditor import _paraphrase_match

        # Below the 6 content-token floor — too noisy to score.
        assert _paraphrase_match("Yes I did", ["Yes I did"]) is None

    def test_withheld_utterance_paraphrase_violation(self):
        """The full ``_withheld_utterance_leak_violations`` path must
        emit a ``major`` violation for paraphrased withheld content."""
        from shadow_loom.auditor import _withheld_utterance_leak_violations

        ws = WorldStateV1(
            locations={
                "LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={},
                ),
            },
            objects={},
            entities={
                "ENT_ALICE": Entity(
                    id="ENT_ALICE", name="Alice",
                    location_id="LOC_A", status="healthy",
                    traits={"trust": TraitVector(value=0.5, inertia=0.2)},
                ),
            },
            events=[
                EventNode(
                    id="EVT_FUTURE",
                    fabula_time=99,
                    syuzhet_index=99,
                    event_type="utterance",
                    actor_ids=["ENT_ALICE"],
                    speaker_id="ENT_ALICE",
                    addressee_ids=[],
                    content=(
                        "I poisoned the chalice that Macbeth drank "
                        "from last night"
                    ),
                    truth_value="true",
                    description="future confession",
                ),
            ],
            causal_topology=[],
        )
        # Reworded paraphrase — no verbatim substring overlap with the
        # canonical sentence above.
        prose = (
            "She had, the night before, slipped a poison into the "
            "very chalice that Macbeth drank from."
        )
        violations = _withheld_utterance_leak_violations(
            prose, ws, syuzhet_anchor=5,
        )
        # At least one paraphrase-flagged major violation.
        assert any(
            v.violation_type == "withheld_utterance_leak"
            and v.severity == "major"
            for v in violations
        ), f"expected paraphrase violation, got: {violations}"


# =====================================================================
# 4. Branch-safe affect / belief / supersession merges
#     (audit follow-up: 6 cross-branch mutation sites)
# =====================================================================
class TestBranchSafeAffectMerges:
    """Shadow merges must not mutate factual-tagged propositions,
    concerns, beliefs, or supersession links \u2014 and vice versa."""

    def _world_with_factual_proposition(self) -> WorldStateV1:
        from shadow_loom.models import Proposition

        return WorldStateV1(
            locations={
                "LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={},
                    world_id="factual",
                ),
            },
            objects={},
            entities={
                "ENT_FACTUAL": Entity(
                    id="ENT_FACTUAL", name="Factual",
                    location_id="LOC_A", status="healthy",
                    traits={"trust": TraitVector(value=0.5, inertia=0.2)},
                    world_id="factual",
                ),
            },
            events=[],
            causal_topology=[],
            propositions=[
                Proposition(
                    proposition_id="PROP_FACT",
                    kind="event_occurs",
                    description="Macbeth is king.",
                    referent_ids=["ENT_FACTUAL"],
                    world_id="factual",
                ),
            ],
        )

    def test_shadow_truth_commit_does_not_mutate_factual_proposition(self, caplog):
        from shadow_loom.extract_graph import (
            MergeChangeset,
            _apply_affect_to_world,
        )
        from shadow_loom.ingestion import (
            ChunkTopology,
            PropositionTruthCommit,
        )

        ws = self._world_with_factual_proposition()
        topology = ChunkTopology(
            chunk_id="CHK_TEST",
            proposition_truth_commits=[
                PropositionTruthCommit(
                    proposition_id="PROP_FACT",
                    fabula_time=10,
                    truth=False,
                    triggered_by="EVT_TEST",
                ),
            ],
        )
        changeset = MergeChangeset()
        with caplog.at_level(logging.INFO):
            _apply_affect_to_world(
                ws, topology,
                world_id="shadow", changeset=changeset,
            )
        # Factual proposition's truth_at_fabula is untouched.
        prop = next(p for p in ws.propositions if p.proposition_id == "PROP_FACT")
        assert 10 not in prop.truth_at_fabula
        assert changeset.proposition_truths_committed == 0
        # INFO skip log emitted.
        assert any(
            "[merge·affect]" in r.getMessage()
            and "PROP_FACT" in r.getMessage()
            and "cross-branch write blocked" in r.getMessage()
            for r in caplog.records
        )

    def test_shadow_belief_confidence_update_skips_factual_entity(self, caplog):
        from shadow_loom.extract_graph import (
            MergeChangeset,
            _apply_belief_confidence_updates,
        )
        from shadow_loom.ingestion import (
            BeliefConfidenceUpdate,
            ChunkTopology,
            EntityUpdate,
        )
        from shadow_loom.models import Belief

        ws = self._world_with_factual_proposition()
        ws.entities["ENT_FACTUAL"].beliefs.append(Belief(
            target_id="OBJ_X",
            perceived_state="X is intact",
            confidence=0.9,
            inertia=0.3,
        ))
        topology = ChunkTopology(
            chunk_id="CHK_TEST",
            entity_updates=[EntityUpdate(
                entity_id="ENT_FACTUAL",
                fabula_time=11,
                belief_confidence_updates=[BeliefConfidenceUpdate(
                    target_id="OBJ_X",
                    new_confidence=0.0,
                )],
            )],
        )
        changeset = MergeChangeset()
        with caplog.at_level(logging.INFO):
            _apply_belief_confidence_updates(
                ws, topology,
                changeset=changeset, merge_world_id="shadow",
            )
        # Factual belief confidence preserved at 0.9.
        b = ws.entities["ENT_FACTUAL"].beliefs[0]
        assert b.confidence == 0.9
        assert changeset.belief_confidence_updates_applied == 0
        assert any(
            "[merge·belief]" in r.getMessage()
            and "ENT_FACTUAL" in r.getMessage()
            for r in caplog.records
        )

    def test_shadow_supersession_does_not_stamp_factual_event(self, caplog):
        from shadow_loom.extract_graph import (
            MergeChangeset,
            _apply_supersession,
        )
        from shadow_loom.ingestion import ChunkTopology

        ws = self._world_with_factual_proposition()
        ws.events.append(EventNode(
            id="EVT_FACTUAL",
            fabula_time=5,
            syuzhet_index=5,
            event_type="outcome",
            description="Macbeth crowned.",
            actor_ids=["ENT_FACTUAL"],
            world_id="factual",
        ))
        topology = ChunkTopology(
            chunk_id="CHK_TEST",
            supersedes_event_ids={"EVT_NEW": "EVT_FACTUAL"},
        )
        changeset = MergeChangeset()
        with caplog.at_level(logging.INFO):
            _apply_supersession(
                ws, topology,
                changeset=changeset, merge_world_id="shadow",
            )
        evt = next(e for e in ws.events if e.id == "EVT_FACTUAL")
        assert evt.superseded_by_event_id is None
        assert changeset.events_superseded == 0
        assert any(
            "[merge·supersede]" in r.getMessage()
            and "EVT_FACTUAL" in r.getMessage()
            for r in caplog.records
        )


# =====================================================================
# 5. Paraphrase normalization edge cases
# =====================================================================
class TestParaphraseNormalization:
    """``_paraphrase_tokens`` must collapse contractions and unicode
    dashes so paraphrase scoring isn't fooled by typography."""

    def test_contractions_collapse_to_stem(self):
        from shadow_loom.auditor import _paraphrase_tokens

        # Apostrophes (ASCII + smart) strip cleanly so "don't" → "dont".
        toks = _paraphrase_tokens("Don\u2019t poison the chalice tonight")
        assert "dont" in toks
        assert "poison" in toks
        assert "chalice" in toks

    def test_unicode_dashes_split_words(self):
        from shadow_loom.auditor import _paraphrase_tokens

        toks = _paraphrase_tokens(
            "the mother\u2014in\u2014law slipped poison into the cup"
        )
        # Em-dashes split the compound at the dash boundary.
        assert "mother" in toks
        assert "law" in toks
        assert "poison" in toks

    def test_smart_quote_paraphrase_still_matches(self):
        from shadow_loom.auditor import _paraphrase_match

        canonical = (
            "I poisoned the chalice that Macbeth drank from last night"
        )
        # Same content reworded with smart quote + em-dash + contraction.
        prose = [
            "She’d poisoned that very chalice last night — the one "
            "Macbeth drank from — with her own hands."
        ]
        match = _paraphrase_match(canonical, prose)
        assert match is not None


# =====================================================================
# 6. promote_branch preserves source branch_label
# =====================================================================
class TestPromoteBranchPreservesProvenance:
    def test_force_promotion_carries_source_branch_label(self):
        from shadow_loom.db import (
            create_project,
            init_db,
            promote_branch,
            save_version,
            upsert_user,
        )
        import json

        init_db("sqlite://")
        user = upsert_user(
            "local", "audit-promote", "audituser",
            email="audit@example.com",
        )
        project = create_project(
            name="audit-promote", owner_id=user.id, description="",
        )
        empty = json.dumps({
            "locations": {}, "objects": {}, "entities": {},
            "events": [], "causal_topology": [],
        })
        # factual root v0
        v0 = save_version(
            project_id=project.id, world_state_json=empty,
            version=0, source="seed", description="root",
            user_id=user.id, world_id="factual",
        )
        # shadow fork off v0 with a label
        shadow = save_version(
            project_id=project.id, world_state_json=empty,
            version=1, ancestor_id=v0.id,
            source="branch_off", description="ctf",
            user_id=user.id, world_id="shadow",
            branch_label="what-if-macbeth-refused",
        )
        promoted = promote_branch(
            shadow.id, user_id=user.id, force=False,
        )
        # branch_label propagated from source.
        assert promoted.branch_label == "what-if-macbeth-refused"
        assert promoted.world_id == "factual"


# =====================================================================
# 7. Branch-safe genesis backfill + entity/object update writes
# =====================================================================
class TestBranchSafeGenesisAndUpdates:
    """A shadow merge whose ``ChunkTopology`` re-emits an id that
    already exists on the factual mainline must NOT mutate the factual
    record \u2014 neither via attribute backfill (``_backfill_*``) nor
    via ``entity_updates`` / ``object_updates`` snapshot appends.
    Mirror for factual merges encountering shadow-tagged holders.
    """

    def _world_with_factual_macbeth(self):
        from shadow_loom.extract_graph import VersionedWorldModel
        from shadow_loom.models import NarrativeObject

        ws = WorldStateV1(
            locations={
                "LOC_CASTLE": Location(
                id="LOC_CASTLE",
                name="Castle", description="Inverness", ambient_state={},
                    world_id="factual",
                ),
            },
            objects={
                "OBJ_DAGGER": NarrativeObject(
                    id="OBJ_DAGGER",
                    name="Dagger",
                    description="A dagger.",
                    location_id="LOC_CASTLE",
                    owner_id=None,
                    properties={"sharp": "true"},
                    affordances=[],
                    world_id="factual",
                ),
            },
            entities={
                "ENT_MACBETH": Entity(
                    id="ENT_MACBETH",
                    name="Macbeth",
                    location_id="LOC_CASTLE",
                    status="healthy",
                    traits={"ambition": TraitVector(value=0.4, inertia=0.3)},
                    world_id="factual",
                ),
            },
            events=[],
            causal_topology=[],
        )
        return VersionedWorldModel.from_world_state(ws)

    def test_shadow_merge_does_not_backfill_factual_entity(self, caplog):
        from shadow_loom.ingestion import ChunkTopology

        vwm = self._world_with_factual_macbeth()
        # Shadow chunk re-emits ENT_MACBETH with extra trait.
        shadow_macbeth = Entity(
            id="ENT_MACBETH",
            name="Macbeth",
            location_id="LOC_CASTLE",
            status="healthy",
            traits={
                "ambition": TraitVector(value=0.4, inertia=0.3),
                "guilt": TraitVector(value=0.9, inertia=0.5),
            },
            world_id="shadow",
        )
        topology = ChunkTopology(
            chunk_id="CHK_SHADOW",
            new_entities={"ENT_MACBETH": shadow_macbeth},
        )
        with caplog.at_level(logging.INFO):
            vwm2 = vwm.merge(topology, world_id="shadow")

        # Factual ENT_MACBETH untouched: no shadow "guilt" trait
        # leaked into the canonical record.
        ent = vwm2.current.entities["ENT_MACBETH"]
        assert ent.world_id == "factual"
        assert "guilt" not in ent.traits
        assert any(
            "[merge\u00b7genesis]" in r.getMessage()
            and "ENT_MACBETH" in r.getMessage()
            for r in caplog.records
        )

    def test_shadow_merge_does_not_backfill_factual_object(self, caplog):
        from shadow_loom.ingestion import ChunkTopology
        from shadow_loom.models import NarrativeObject

        vwm = self._world_with_factual_macbeth()
        shadow_dagger = NarrativeObject(
            id="OBJ_DAGGER",
            name="Dagger",
            description="A poisoned dagger.",
            location_id="LOC_CASTLE",
            owner_id=None,
            properties={"sharp": "true", "poisoned": "true"},
            affordances=[],
            world_id="shadow",
        )
        topology = ChunkTopology(
            chunk_id="CHK_SHADOW",
            new_objects={"OBJ_DAGGER": shadow_dagger},
        )
        with caplog.at_level(logging.INFO):
            vwm2 = vwm.merge(topology, world_id="shadow")
        obj = vwm2.current.objects["OBJ_DAGGER"]
        assert obj.world_id == "factual"
        assert "poisoned" not in obj.properties
        assert any(
            "[merge\u00b7genesis]" in r.getMessage()
            and "OBJ_DAGGER" in r.getMessage()
            for r in caplog.records
        )

    def test_shadow_merge_skips_entity_update_on_factual_holder(self, caplog):
        from shadow_loom.ingestion import ChunkTopology, EntityUpdate

        vwm = self._world_with_factual_macbeth()
        eu = EntityUpdate(
            entity_id="ENT_MACBETH",
            fabula_time=10,
            triggered_by=None,
            trait_updates={
                "guilt": TraitVector(value=0.95, inertia=0.5),
            },
            new_status="ill",
        )
        topology = ChunkTopology(
            chunk_id="CHK_SHADOW",
            entity_updates=[eu],
        )
        with caplog.at_level(logging.INFO):
            vwm2 = vwm.merge(topology, world_id="shadow")
        ent = vwm2.current.entities["ENT_MACBETH"]
        # Factual timeline untouched: the shadow snapshot was routed
        # into a synthesized ``shadow-orphan-*`` sidecar clone, not
        # appended onto the factual holder.
        assert ent.state_timeline == []
        # Replay on the factual holder still returns the baseline.
        from shadow_loom.models import reconstruct_entity_at
        recon = reconstruct_entity_at(ent, fabula_time=20)
        assert "guilt" not in recon["traits"]
        assert recon["status"] == "healthy"
        # Synthesized sidecar captured the shadow write.
        assert vwm2.current.shadow_entities
        synth_label = next(iter(vwm2.current.shadow_entities))
        assert synth_label.startswith("shadow-orphan")
        assert "ENT_MACBETH" in vwm2.current.shadow_entities[synth_label]

    def test_shadow_merge_skips_object_update_on_factual_holder(self, caplog):
        from shadow_loom.ingestion import ChunkTopology, ObjectUpdate

        vwm = self._world_with_factual_macbeth()
        ou = ObjectUpdate(
            object_id="OBJ_DAGGER",
            fabula_time=12,
            triggered_by=None,
            properties_set={"poisoned": "true"},
        )
        topology = ChunkTopology(
            chunk_id="CHK_SHADOW",
            object_updates=[ou],
        )
        with caplog.at_level(logging.INFO):
            vwm2 = vwm.merge(topology, world_id="shadow")
        obj = vwm2.current.objects["OBJ_DAGGER"]
        # Factual object timeline untouched: shadow write routed into
        # synthesized sidecar.
        assert obj.state_timeline == []
        from shadow_loom.models import reconstruct_object_at
        recon = reconstruct_object_at(obj, fabula_time=20)
        assert "poisoned" not in recon["properties"]
        assert vwm2.current.shadow_objects
        synth_label = next(iter(vwm2.current.shadow_objects))
        assert synth_label.startswith("shadow-orphan")
        assert "OBJ_DAGGER" in vwm2.current.shadow_objects[synth_label]

    def test_reconstruct_entity_at_filters_cross_branch_snapshots(self):
        """Defensive replay-side guard: even if a polluted timeline
        survives in persisted data, ``reconstruct_entity_at`` skips
        snapshots tagged with a different branch than the holder."""
        from shadow_loom.models import (
            EntityStateSnapshot,
            reconstruct_entity_at,
        )

        ent = Entity(
            id="ENT_MACBETH",
            name="Macbeth",
            location_id="LOC_CASTLE",
            status="healthy",
            traits={"ambition": TraitVector(value=0.4, inertia=0.3)},
            world_id="factual",
        )
        # A legacy / polluted shadow snapshot on a factual holder.
        ent.state_timeline.append(EntityStateSnapshot(
            world_id="shadow",
            fabula_time=10,
            triggered_by=None,
            traits={"guilt": TraitVector(value=0.95, inertia=0.5)},
            status="ill",
        ))
        # And a clean factual snapshot at the same tick.
        ent.state_timeline.append(EntityStateSnapshot(
            world_id="factual",
            fabula_time=10,
            triggered_by=None,
            traits={"ambition": TraitVector(value=0.6, inertia=0.3)},
        ))
        recon = reconstruct_entity_at(ent, fabula_time=20)
        # Shadow snapshot ignored.
        assert "guilt" not in recon["traits"]
        assert recon["status"] == "healthy"
        # Factual one applied.
        assert recon["traits"]["ambition"]["value"] == pytest.approx(0.6)
