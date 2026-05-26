# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the round-3 (and surviving round-2) audit
remediation batch dated 2026-05-26.

Each section pins a single audit finding so regressions in the
auditor configuration surface, the brief formatters, the ingestion
quality gates, and the surprise/mystery scorers fail loudly. See
``/memories/repo/round3-deep-audit-2026-05-25.md`` for the audit
report this file pins.
"""
from __future__ import annotations

import logging
from typing import Optional
from unittest.mock import MagicMock

import pytest

from shadow_loom import auditor as auditor_mod
from shadow_loom import generation as generation_mod
from shadow_loom import ingestion as ingestion_mod
from shadow_loom.auditor import (
    _KNOWN_TARGET_EFFECTS,
    UNIVERSAL_AUDIT_CATEGORIES,
    resolve_audit_categories,
)
from shadow_loom.generation import _SCENIC_MODES, _format_counterfactual
from shadow_loom.ingestion import (
    Channel,
    ConsequencesExtraction,
    EntityUpdate,
    PhysicsExtraction,
    SocialExtraction,
    _consequences_mutation_parity_broken,
    _social_channel_underextracted,
)
from shadow_loom.models import (
    CausalEdge,
    EventNode,
)


# ---------------------------------------------------------------------------
# Round-2 #8 — _KNOWN_TARGET_EFFECTS / resolve_audit_categories surface
# ---------------------------------------------------------------------------

class TestKnownTargetEffects:
    def test_core_directive_effects_listed(self):
        for eff in (
            "mystery", "dramatic_irony", "surprise", "suspense",
            "fear", "joy", "regret", "grief", "rage", "love",
            "narrative_tension",
        ):
            assert eff in _KNOWN_TARGET_EFFECTS, eff

    def test_non_directive_effects_listed(self):
        for eff in (
            "observation", "intervention", "counterfactual",
            "general", "interrogate",
        ):
            assert eff in _KNOWN_TARGET_EFFECTS

    def test_unknown_effect_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=auditor_mod.logger.name):
            cats = resolve_audit_categories("typo_effect_xyz")
        # Falls back to physics + universals only.
        assert "physics" in cats
        for u in UNIVERSAL_AUDIT_CATEGORIES:
            assert u in cats
        # And the warning surfaced.
        assert any(
            "unknown" in r.message and "typo_effect_xyz" in r.message
            for r in caplog.records
        )

    def test_known_effect_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger=auditor_mod.logger.name):
            cats = resolve_audit_categories("mystery")
        assert "physics" in cats
        # narrative effects pull in epistemic/probabilistic
        assert "epistemic" in cats
        assert not any(
            "unknown target_effect" in r.message for r in caplog.records
        )

    def test_narrative_tension_resolves_to_full_category_set(self):
        cats = resolve_audit_categories("narrative_tension")
        for required in ("epistemic", "probabilistic", "physics"):
            assert required in cats
        for u in UNIVERSAL_AUDIT_CATEGORIES:
            assert u in cats
        # No duplicates.
        assert len(cats) == len(set(cats))


# ---------------------------------------------------------------------------
# Round-2 #4 — _SCENIC_MODES contains narrative_tension
# ---------------------------------------------------------------------------

class TestScenicModes:
    def test_narrative_tension_is_scenic(self):
        assert "narrative_tension" in _SCENIC_MODES

    def test_core_scenic_modes_present(self):
        for m in (
            "observation", "intervention", "counterfactual",
            "default", "fallback",
        ):
            assert m in _SCENIC_MODES


# ---------------------------------------------------------------------------
# Round-3 R3 — _MINOR_BYPASS_ALLOWLIST pruned to {"style_mismatch"}
# ---------------------------------------------------------------------------

class TestMinorBypassAllowlist:
    """The dead cosmetic-bypass values (``pacing``, ``diction``,
    ``tone``, etc.) were never wired through ``AuditViolation.
    violation_type`` and were pruned. Reading the source string is
    the most reliable way to assert this without executing the loop.
    """

    def test_only_style_mismatch_remains(self):
        import inspect
        from shadow_loom import auditor as a
        src = inspect.getsource(a)
        # The allowlist literal is uniquely identifiable.
        marker = "_MINOR_BYPASS_ALLOWLIST = {"
        idx = src.find(marker)
        assert idx >= 0, "allowlist literal moved"
        end = src.find("}", idx)
        body = src[idx + len(marker):end]
        # Strip comments and whitespace; collect string literals.
        literals = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.split(","):
                token = token.strip().strip(",")
                # Strip trailing comment on same line.
                if "#" in token:
                    token = token.split("#", 1)[0].strip()
                if token.startswith('"') and token.endswith('"'):
                    literals.append(token[1:-1])
                elif token.startswith("'") and token.endswith("'"):
                    literals.append(token[1:-1])
        assert literals == ["style_mismatch"], literals


# ---------------------------------------------------------------------------
# Round-2 #7 — mutation_social parity covers both dyad endpoints
# ---------------------------------------------------------------------------

class TestMutationSocialParity:
    def test_missing_counterpart_flagged(self):
        physics = PhysicsExtraction(
            events=[
                EventNode(
                    id="EVT_QUARREL",
                    fabula_time=10,
                    syuzhet_index=10,
                    event_type="outcome",
                    actor_ids=["ENT_ALICE"],
                    target_ids=["ENT_BOB"],
                    description="quarrel",
                ),
            ],
            causal_topology=[
                CausalEdge(
                    source_id="EVT_QUARREL",
                    target_id="ENT_ALICE",
                    causality_type="mutation_social",
                    causal_force=0.6,
                    mechanism="psychological",
                    fabula_time=10,
                    rel_counterpart_id="ENT_BOB",
                ),
            ],
        )
        # Only Alice gets an EntityUpdate; Bob is missing.
        consequences = ConsequencesExtraction(
            entity_updates=[
                EntityUpdate(
                    entity_id="ENT_ALICE",
                    fabula_time=10,
                    triggered_by="EVT_QUARREL",
                ),
            ],
        )
        missing = _consequences_mutation_parity_broken(physics, consequences)
        assert missing == ["ENT_BOB"]

    def test_both_endpoints_present_returns_empty(self):
        physics = PhysicsExtraction(
            causal_topology=[
                CausalEdge(
                    source_id="EVT_QUARREL",
                    target_id="ENT_ALICE",
                    causality_type="mutation_social",
                    causal_force=0.6,
                    mechanism="psychological",
                    fabula_time=10,
                    rel_counterpart_id="ENT_BOB",
                ),
            ],
        )
        consequences = ConsequencesExtraction(
            entity_updates=[
                EntityUpdate(
                    entity_id="ENT_ALICE", fabula_time=10,
                    triggered_by="EVT_QUARREL",
                ),
                EntityUpdate(
                    entity_id="ENT_BOB", fabula_time=10,
                    triggered_by="EVT_QUARREL",
                ),
            ],
        )
        assert _consequences_mutation_parity_broken(physics, consequences) == []


# ---------------------------------------------------------------------------
# Round-3 #7 — channel-underextraction respects prior_channels
# ---------------------------------------------------------------------------

class TestChannelUnderextractionPriorAware:
    def _two_utterances(self) -> SocialExtraction:
        return SocialExtraction(
            utterance_events=[
                EventNode(
                    id="EVT_U1", fabula_time=1, syuzhet_index=1,
                    event_type="utterance",
                    actor_ids=["ENT_ALICE"], target_ids=["ENT_BOB"],
                    speaker_id="ENT_ALICE",
                    addressee_ids=["ENT_BOB"],
                    description="letter 1",
                ),
                EventNode(
                    id="EVT_U2", fabula_time=2, syuzhet_index=2,
                    event_type="utterance",
                    actor_ids=["ENT_ALICE"], target_ids=["ENT_BOB"],
                    speaker_id="ENT_ALICE",
                    addressee_ids=["ENT_BOB"],
                    description="letter 2",
                ),
            ],
        )

    def test_fires_without_any_channel(self):
        assert _social_channel_underextracted(self._two_utterances()) is True

    def test_suppressed_by_prior_channel_covering_pair(self):
        social = self._two_utterances()
        prior = {
            "CHN_LETTERS": Channel(
                id="CHN_LETTERS",
                name="letter correspondence",
                medium="letter",
                participant_ids=["ENT_ALICE", "ENT_BOB"],
            ),
        }
        assert _social_channel_underextracted(
            social, prior_channels=prior,
        ) is False

    def test_unrelated_prior_channel_does_not_suppress(self):
        social = self._two_utterances()
        prior = {
            "CHN_PHONE": Channel(
                id="CHN_PHONE",
                name="phone line",
                medium="telephone",
                participant_ids=["ENT_CAROL", "ENT_DAVE"],
            ),
        }
        assert _social_channel_underextracted(
            social, prior_channels=prior,
        ) is True


# ---------------------------------------------------------------------------
# Round-3 #6 — channel CF formatter handles active=None as unknown
# ---------------------------------------------------------------------------

def _make_cf(do_target) -> MagicMock:
    cf = MagicMock()
    cf.actual_outcome = "what happened"
    cf.simulated_outcome = "what could have"
    cf.divergence_event_id = None
    cf.divergence_description = None
    cf.do_target = do_target
    cf.do_target_gloss = None
    cf.do_target_context = ""
    cf.affected_propositions = []
    cf.affected_proposition_descriptions = []
    cf.affected_beliefs = []
    cf.affected_belief_descriptions = []
    cf.affected_relationships = []
    cf.affected_emotions = []
    cf.salience_score = 0.0
    cf.regret_signal = 0.0
    return cf


class TestCounterfactualChannelActiveNone:
    def test_active_none_renders_as_unknown(self):
        do_target = MagicMock()
        do_target.target_kind = "channel"
        do_target.channel_id = "CHN_X"
        do_target.active = None
        cf = _make_cf(do_target)
        out = _format_counterfactual(cf)
        assert "severed" not in out
        assert "transmissibility" in out.lower()

    def test_active_true_still_renders_as_open(self):
        do_target = MagicMock()
        do_target.target_kind = "channel"
        do_target.channel_id = "CHN_X"
        do_target.active = True
        cf = _make_cf(do_target)
        out = _format_counterfactual(cf)
        assert "open" in out

    def test_active_false_still_renders_as_severed(self):
        do_target = MagicMock()
        do_target.target_kind = "channel"
        do_target.channel_id = "CHN_X"
        do_target.active = False
        cf = _make_cf(do_target)
        out = _format_counterfactual(cf)
        assert "severed" in out


# ---------------------------------------------------------------------------
# Round-3 #5 — incomplete typed do_target payload emits explicit fallback
# ---------------------------------------------------------------------------

class TestCounterfactualIncompletePayloadFallback:
    def test_incomplete_proposition_target_emits_fallback(self):
        do_target = MagicMock()
        do_target.target_kind = "proposition"
        # Missing pid and truth → branch silently dropped previously.
        do_target.proposition_id = None
        do_target.truth = None
        cf = _make_cf(do_target)
        out = _format_counterfactual(cf)
        assert "RUNG-3 SURGERY KIND: proposition" in out
        assert "incomplete do_target payload" in out

    def test_unknown_kind_emits_fallback(self):
        do_target = MagicMock()
        do_target.target_kind = "made_up_kind"
        cf = _make_cf(do_target)
        out = _format_counterfactual(cf)
        assert "RUNG-3 SURGERY KIND: made_up_kind" in out
        assert "incomplete do_target payload" in out

    def test_complete_payload_does_not_emit_fallback(self):
        do_target = MagicMock()
        do_target.target_kind = "proposition"
        do_target.proposition_id = "PROP_X"
        do_target.truth = True
        cf = _make_cf(do_target)
        out = _format_counterfactual(cf)
        assert "RUNG-3 SURGERY KIND: proposition" in out
        assert "incomplete do_target payload" not in out


# ---------------------------------------------------------------------------
# Round-3 #1 — _top_mystery_questions accepts syuzhet_anchor
# ---------------------------------------------------------------------------

class TestMysteryAnchorPlumbing:
    def test_signature_accepts_syuzhet_anchor_kwarg(self):
        import inspect
        from shadow_loom.directive_assembly import DirectiveAssembler
        sig = inspect.signature(DirectiveAssembler._top_mystery_questions)
        assert "syuzhet_anchor" in sig.parameters
        # Default is None so legacy callers keep working.
        assert sig.parameters["syuzhet_anchor"].default is None


# ---------------------------------------------------------------------------
# Round-3 #2 — surprise scorer source-side branch uses event actor_ids
# ---------------------------------------------------------------------------

class TestSurpriseSourceSideReachable:
    """The unreachable ``ce.source_id == eid`` branch was replaced
    with membership against the source event's ``actor_ids``. The
    most lightweight assertion is that the new source-actor map is
    actually constructed in the live source, which we verify by
    inspecting the source for the new lookup.
    """

    def test_event_actor_lookup_is_built(self):
        import inspect
        from shadow_loom.directive_assembly import DirectiveAssembler
        src = inspect.getsource(DirectiveAssembler.compute_surprise_score)
        # New lookup table for source-actor membership.
        assert "_evt_actors" in src
        # Membership test threaded through the source-side branch.
        assert "_evt_actors.get(ce.source_id" in src
        # The old broken branch no longer present.
        assert "elif ce.source_id == eid:" not in src


# ---------------------------------------------------------------------------
# Round-3 #10 — world_trait_shifts surfaced to renderer prompt
# ---------------------------------------------------------------------------

class TestWorldTraitShiftsInRendererPrompt:
    """The renderer prompt builder must now contain a world-trait
    section. We inspect the source rather than running the full
    pipeline to keep this test offline-safe.
    """

    def test_renderer_prompt_section_exists(self):
        import inspect
        src = inspect.getsource(generation_mod)
        assert "WORLD-TRAIT SHIFTS" in src
        assert "world_trait_shifts" in src
