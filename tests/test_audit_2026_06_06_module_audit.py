# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-06-06 deeper module audit.

Three independent findings from a static review of modules not covered
by the earlier timeline audit:

  1. ``AMWNInstantiator.create_sandbox`` computed the spatial-lifecycle
     frontier (``_spatial_max_ft``) from ``recent_memory`` only. The
     ``generation.py`` payload builder splits utterance events OUT of
     ``recent_memory`` into ``relevant_utterance_events``, whereas
     ``extract_graph`` leaves them in. So on the generation path, an
     utterance-latest slice understated the frontier — silently
     disabling the "drop passages destroyed before now" gate and
     stamping ambient edges at too-early a fabula_time.

  2. ``query_parsing`` rendered the proposition prompt context using
     ``prop.content`` / ``prop.referenced_node_ids`` — fields that do
     not exist on :class:`Proposition` (the real fields are
     ``description`` / ``referent_ids``). ``getattr(..., default)``
     masked the mismatch, so every PROP_ line rendered blank.

  3. ``_concern_window_at`` could return a malformed (length != 2)
     ``activation_fabula_window`` straight to a callsite that unpacked
     ``lo, hi = win`` without a length guard, crashing on bad input.
"""
from __future__ import annotations

import importlib

from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.query_parsing import _build_graph_summary, _format_valid_ids_section
from shadow_loom.affect_unification import _concern_window_at
from shadow_loom.models import Concern


# ---------------------------------------------------------------------------
# 1. Instantiator spatial frontier must include utterance events.
# ---------------------------------------------------------------------------
class TestSpatialFrontierIncludesUtterances:

    @staticmethod
    def _payload(recent_memory, utterance_events):
        return {
            "current_locations": [
                {"id": "LOC_A", "name": "A"},
                {"id": "LOC_B", "name": "B"},
            ],
            "focus_entities": [],
            "present_entities": [],
            "present_objects": [],
            "recent_memory": recent_memory,
            "relevant_utterance_events": utterance_events,
            "relevant_spatial_edges": [
                # Passage destroyed at t=8.
                {"source_id": "LOC_A", "target_id": "LOC_B",
                 "established_at_fabula": 1, "destroyed_at_fabula": 8},
            ],
        }

    def _has_passage(self, sandbox) -> bool:
        return any(
            d.get("edge_type") == "connected_to"
            for _, _, d in sandbox.edges(data=True)
        )

    def test_generation_path_utterance_latest_drops_destroyed_passage(self):
        # Generation path: utterances are stripped from recent_memory.
        # Latest known moment is the utterance at t=10, so a passage
        # destroyed at t=8 must NOT survive.
        payload = self._payload(
            recent_memory=[],
            utterance_events=[{"id": "EVT_U", "fabula_time": 10,
                               "event_type": "utterance"}],
        )
        sandbox = AMWNInstantiator.create_sandbox(payload, "observation")
        assert not self._has_passage(sandbox), (
            "passage destroyed at t=8 leaked because the frontier ignored "
            "the t=10 utterance"
        )

    def test_extract_graph_path_recent_memory_still_works(self):
        # extract_graph path: the same event lives in recent_memory.
        payload = self._payload(
            recent_memory=[{"id": "EVT_U", "fabula_time": 10,
                            "event_type": "utterance"}],
            utterance_events=[],
        )
        sandbox = AMWNInstantiator.create_sandbox(payload, "observation")
        assert not self._has_passage(sandbox)

    def test_passage_survives_when_destroyed_after_frontier(self):
        # Frontier at t=5 (utterance), passage destroyed later at t=8 ->
        # still active, must survive.
        payload = self._payload(
            recent_memory=[],
            utterance_events=[{"id": "EVT_U", "fabula_time": 5,
                               "event_type": "utterance"}],
        )
        sandbox = AMWNInstantiator.create_sandbox(payload, "observation")
        assert self._has_passage(sandbox)


# ---------------------------------------------------------------------------
# 2. Proposition prompt context must render real fields.
# ---------------------------------------------------------------------------
class TestPropositionPromptFields:

    def test_graph_summary_renders_description_and_referents(self):
        ws = importlib.import_module("example_worlds.macbeth").world_state
        assert ws.propositions, "fixture must have propositions"
        out = _build_graph_summary(ws)
        p = ws.propositions[0]
        assert p.description[:40] in out
        # at least one referent id should appear in the refs=[...] block
        if p.referent_ids:
            assert p.referent_ids[0] in out

    def test_valid_ids_section_renders_description(self):
        ws = importlib.import_module("example_worlds.macbeth").world_state
        out = _format_valid_ids_section(ws)
        p = ws.propositions[0]
        assert p.description[:40] in out


# ---------------------------------------------------------------------------
# 3. _concern_window_at normalises / rejects malformed windows.
# ---------------------------------------------------------------------------
class TestConcernWindowNormalisation:

    @staticmethod
    def _concern(window):
        return Concern(
            concern_id="CCN_T", proposition_id="PROP_T", polarity="fear",
            activation_fabula_window=window,
        )

    def test_wellformed_window_returns_int_tuple(self):
        assert _concern_window_at(self._concern([2, 9]), 5) == (2, 9)

    def test_none_window_returns_none(self):
        assert _concern_window_at(self._concern(None), 5) is None

    def test_malformed_short_window_returns_none_not_crash(self):
        assert _concern_window_at(self._concern([3]), 5) is None

    def test_malformed_long_window_coerces_first_two(self):
        # length-3 used to crash `lo, hi = win`; now first two are taken.
        assert _concern_window_at(self._concern([1, 4, 9]), 5) == (1, 4)
