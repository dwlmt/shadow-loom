# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the round-4 audit remediation batch dated
2026-05-26.

Each section pins one finding from the round-4 audit so a regression
in the typed-DO dispatcher, the sandbox/canonical ordering, the
spatial tombstone semantics, the legacy interventions envelope, the
remap-on-feedback helper, the MCP error sanitiser, or the project
name-resolution authorisation will fail loudly.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import networkx as nx
import pytest

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import (
    Entity,
    EventNode,
    Location,
    RelationshipEdge,
    RelationshipMetric,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.query_models import (
    DoChannel,
    DoCausalEdge,
    DoRelationship,
    DoSpatialEdge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_entity_world() -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_A": Location(
                id="LOC_A", name="A", description="A", ambient_state={},
            ),
            "LOC_B": Location(
                id="LOC_B", name="B", description="B", ambient_state={},
            ),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.2)},
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.7, inertia=0.3)},
            ),
        },
        events=[
            EventNode(
                id="EVT_MEET", fabula_time=1, syuzhet_index=1,
                event_type="choice", actor_ids=["ENT_ALICE"],
                target_ids=["ENT_BOB"], description="They meet",
            ),
        ],
        social_topology=[
            RelationshipEdge(
                source_entity_id="ENT_ALICE",
                target_entity_id="ENT_BOB",
                metrics={
                    "affinity": RelationshipMetric(
                        value=0.2, inertia=0.3, observed=True,
                        last_updated_fabula=1,
                    ),
                },
                established_at_fabula=1,
            ),
        ],
        spatial_topology=[
            SpatialEdge(
                source_id="LOC_A", target_id="LOC_B",
                connection_type="doorway", bidirectional=True,
                established_at_fabula=0,
            ),
        ],
        causal_topology=[],
    )


def _engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE", "ENT_BOB"])
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    return CausalPhysicsEngine(sandbox, ws)


# ---------------------------------------------------------------------------
# #1 — _apply_do_relationship registers (src, tgt, metric) in
#       _intervened_relationships
# ---------------------------------------------------------------------------

class TestDoRelationshipPinsDyad:
    def test_dyad_triple_pinned(self):
        ws = _two_entity_world()
        eng = _engine(ws)
        eng.apply_do_targets([
            DoRelationship(
                source_entity_id="ENT_ALICE",
                target_entity_id="ENT_BOB",
                metric="affinity",
                value=-0.9,
                fabula_time=2,
            ),
        ])
        assert ("ENT_ALICE", "ENT_BOB", "affinity") in eng._intervened_relationships


# ---------------------------------------------------------------------------
# #2 — _apply_do_relationship fails closed when sandbox lacks endpoints
# ---------------------------------------------------------------------------

class TestDoRelationshipFailsClosedOnMissingSandboxEndpoints:
    def test_no_canonical_write_when_sandbox_missing(self):
        ws = _two_entity_world()
        eng = _engine(ws)
        # Strip the endpoints out of the sandbox so the
        # ``sb.has_node(...)`` guard inside ``_apply_do_relationship``
        # fails, exercising the fail-closed branch.
        for nid in ("ENT_ALICE", "ENT_BOB"):
            if eng.sandbox.has_node(nid):
                eng.sandbox.remove_node(nid)

        before_counter = getattr(eng, "_edge_do_targets_applied", 0)
        before_rel = next(
            (r for r in (ws.social_topology or [])
             if r.source_entity_id == "ENT_ALICE"
             and r.target_entity_id == "ENT_BOB"),
            None,
        )
        before_value = before_rel.metrics["affinity"].value if before_rel else None

        eng.apply_do_targets([
            DoRelationship(
                source_entity_id="ENT_ALICE",
                target_entity_id="ENT_BOB",
                metric="affinity",
                value=-0.99,
                fabula_time=2,
            ),
        ])

        # No application credit.
        assert eng._edge_do_targets_applied == before_counter
        # Canonical metric is unchanged (the early return prevents
        # the world-state mirror from clamping it).
        after_rel = next(
            r for r in (ws.social_topology or [])
            if r.source_entity_id == "ENT_ALICE"
            and r.target_entity_id == "ENT_BOB"
        )
        assert after_rel.metrics["affinity"].value == before_value


# ---------------------------------------------------------------------------
# #3 — _apply_do_causal_edge validates canonical endpoints BEFORE the
#       sandbox add
# ---------------------------------------------------------------------------

class TestDoCausalEdgeValidatesBeforeSandboxMutation:
    def test_validation_textually_precedes_sandbox_add(self):
        src = inspect.getsource(CausalPhysicsEngine._apply_do_causal_edge)
        validate_idx = src.find("_world_knows_node")
        add_edge_idx = src.find("self.sandbox.add_edge(")
        assert validate_idx != -1 and add_edge_idx != -1
        assert validate_idx < add_edge_idx, (
            "Canonical endpoint validation must run before sandbox mutation "
            "to avoid phantom sandbox edges (round-4 audit fix)."
        )

    def test_dangling_endpoint_does_not_mutate_sandbox(self):
        ws = _two_entity_world()
        eng = _engine(ws)
        before = getattr(eng, "_edge_do_targets_applied", 0)
        eng.apply_do_targets([
            DoCausalEdge(
                source_id="EVT_MEET",
                target_id="EVT_DOES_NOT_EXIST",
                action="add",
                causality_type="chain_reaction",
                mechanism="physical",
            ),
        ])
        # Refused — no count, no sandbox edge.
        assert eng._edge_do_targets_applied == before
        if eng.sandbox.has_node("EVT_MEET"):
            assert not eng.sandbox.has_edge("EVT_MEET", "EVT_DOES_NOT_EXIST")


# ---------------------------------------------------------------------------
# #4 — _apply_do_spatial_edge sever preserves tombstones in
#       world_state.spatial_topology
# ---------------------------------------------------------------------------

class TestDoSpatialEdgeSeverPreservesTombstone:
    def test_severed_edge_stays_with_destroyed_at_fabula(self):
        ws = _two_entity_world()
        eng = _engine(ws)
        eng.apply_do_targets([
            DoSpatialEdge(
                source_id="LOC_A",
                target_id="LOC_B",
                action="sever",
                fabula_time=7,
            ),
        ])
        matches = [
            e for e in (ws.spatial_topology or [])
            if e.source_id == "LOC_A" and e.target_id == "LOC_B"
        ]
        assert matches, (
            "Severed edge must stay in spatial_topology so the "
            "destroyed_at_fabula tombstone survives (round-4 fix)."
        )
        assert matches[0].destroyed_at_fabula == 7


# ---------------------------------------------------------------------------
# #5 — DoChannel legacy mirror surfaces channel.active for provenance pruning
# ---------------------------------------------------------------------------

class TestDoChannelLegacyMirror:
    def test_channel_deactivation_lands_on_legacy_envelope(self):
        ws = _two_entity_world()
        eng = _engine(ws)
        eng.apply_do_targets([
            DoChannel(channel_id="CHN_FAKE", active=False, fabula_time=3),
        ])
        legacy = getattr(eng, "_last_legacy_interventions", {}) or {}
        assert legacy.get("CHN_FAKE.active") is False, (
            "DoChannel(active=False) must land on the legacy "
            "interventions envelope so _collect_provenance_invalidations "
            "can prune utterance provenance (round-4 fix)."
        )


# ---------------------------------------------------------------------------
# #6 — query_parsing._remap_dict_keys remaps dotted-path keys by base ID
# ---------------------------------------------------------------------------

class TestRemapDictKeysHandlesDottedPaths:
    def test_dotted_path_remap_via_fuzzy_repair(self):
        """End-to-end: stage a validation error that names a bad ENT_ id
        used as a dotted-intervention-key prefix and assert the fuzzy
        repair rewrites the prefix on the dotted key, not just the bare
        id."""
        from shadow_loom.models import (
            Entity,
            Location,
            TraitVector,
            WorldStateV1,
        )
        from shadow_loom.query_parsing import (
            ParsedQuery,
            ValidationError,
            _try_fuzzy_repair,
        )

        ws = WorldStateV1(
            locations={
                "LOC_X": Location(
                    id="LOC_X", name="X", description="X", ambient_state={},
                ),
            },
            objects={},
            entities={
                "ENT_GOODNAME": Entity(
                    id="ENT_GOODNAME", name="GoodName",
                    location_id="LOC_X", status="healthy",
                    traits={"fear": TraitVector(value=0.0, inertia=0.3)},
                ),
            },
            events=[], causal_topology=[], spatial_topology=[],
            social_topology=[],
        )

        # ``ENT_GOODNAM`` is a one-char typo of ``ENT_GOODNAME`` —
        # well within the fuzzy threshold. We use it both as a bare key
        # and as the prefix of a dotted intervention key.
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="probe",
            interventions={
                "ENT_GOODNAM.traits.fear": 0.9,
                "ENT_GOODNAM": "bare-key",
            },
        )
        errors = [
            ValidationError(
                field="interventions",
                message="ID 'ENT_GOODNAM' not found in world model.",
            ),
        ]
        repaired, remap, remaining = _try_fuzzy_repair(parsed, errors, ws)

        assert remap.get("ENT_GOODNAM") == "ENT_GOODNAME"
        keys = set((repaired.interventions or {}).keys())
        assert "ENT_GOODNAME.traits.fear" in keys, (
            "Dotted intervention keys must remap on their base entity "
            "ID so feedback-driven ID corrections actually take effect "
            "(round-4 fix)."
        )
        assert "ENT_GOODNAME" in keys
        assert "ENT_GOODNAM.traits.fear" not in keys
        assert "ENT_GOODNAM" not in keys


# ---------------------------------------------------------------------------
# #7 — patch_world_state uses _sanitised_error at all caught-exception sites
# ---------------------------------------------------------------------------

class TestPatchWorldStateUsesSanitisedError:
    def test_source_uses_sanitised_helper(self):
        from shadow_loom_mcp import server as srv

        src = inspect.getsource(srv.patch_world_state)
        # No raw provider/exception fragments leaking through f-strings.
        assert "Invalid patch payload:" not in src
        assert "Patch application failed:" not in src
        assert "Persisting patched world failed:" not in src
        # Sanitised helper is invoked at every catch site (3 expected).
        sanitised_count = src.count('_sanitised_error("patch_world_state"')
        assert sanitised_count >= 3, (
            f"Expected ≥3 _sanitised_error(\"patch_world_state\", ...) "
            f"call sites in patch_world_state body, found {sanitised_count}."
        )


# ---------------------------------------------------------------------------
# #8 — resolve_project falls back through check_project_access for
#       collaborators / admins while still hiding non-accessible projects
# ---------------------------------------------------------------------------

class TestResolveProjectNameFallbackHonoursCollaborators:
    def _setup(self, monkeypatch, *, owned_match, public_match, access_ok):
        from shadow_loom_mcp import helpers as hp

        owned = MagicMock(id=42, is_public=False, owner_id=None)
        global_proj = MagicMock(id=99, is_public=False, owner_id=999)

        def fake_find(name, owner_id=None):
            if owner_id is not None:
                return owned if owned_match else None
            return global_proj if public_match else None

        def fake_access(pid, ctx, *, min_role="viewer"):
            return None if access_ok else "permission denied"

        monkeypatch.setattr(hp, "find_project_by_name", fake_find)
        monkeypatch.setattr(hp, "check_project_access", fake_access)
        monkeypatch.setattr(hp, "get_user_id", lambda ctx: 7)
        return hp

    def test_collaborator_can_resolve_private_project_by_name(self, monkeypatch):
        hp = self._setup(
            monkeypatch,
            owned_match=False,
            public_match=True,
            access_ok=True,
        )
        pid, err = hp.resolve_project(None, "shared-project", ctx=MagicMock())
        assert err is None
        assert pid == 99

    def test_non_member_gets_generic_not_found(self, monkeypatch):
        hp = self._setup(
            monkeypatch,
            owned_match=False,
            public_match=True,
            access_ok=False,
        )
        pid, err = hp.resolve_project(None, "secret-project", ctx=MagicMock())
        assert pid is None
        assert err is not None
        # Existence leak protection: must NOT reveal the access verdict.
        assert "permission" not in err.lower()
        assert "not found" in err.lower()

    def test_owned_match_short_circuits(self, monkeypatch):
        hp = self._setup(
            monkeypatch,
            owned_match=True,
            public_match=False,
            access_ok=True,
        )
        pid, err = hp.resolve_project(None, "my-project", ctx=MagicMock())
        assert err is None
        assert pid == 42
