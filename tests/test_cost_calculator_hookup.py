# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the price-calculator hookup.

Verifies that:

1. :func:`shadow_loom.db.ensure_default_cost_rule` seeds a single
   catch-all ``CostRuleRow`` on ``init_db()`` (input $0.039 / 1M
   tokens, output $0.18 / 1M tokens) and that re-running is
   idempotent.
2. :meth:`CostCalculator.get_cost_rule` falls back to that default
   rule when no provider-specific rule exists, so every logged
   agent call lands on a non-zero unit price.
3. :func:`increment_user_lifetime_usage` upserts onto the
   ``UserUsageSummaryRow(period_type='lifetime')`` row for both the
   built-in example user and arbitrary "other" users, with
   counters incrementing monotonically across calls.
"""
from __future__ import annotations

import pytest
from sqlmodel import select

from shadow_loom.db import (
    DEFAULT_COST_RULE_PROVIDER,
    DEFAULT_INPUT_COST_PER_TOKEN_USD,
    DEFAULT_OUTPUT_COST_PER_TOKEN_USD,
    AgentCallLogRow,
    ApiCallLogRow,
    CostRuleRow,
    UserRow,
    UserUsageSummaryRow,
    ensure_default_cost_rule,
    ensure_example_user,
    get_session,
    init_db,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    yield


class TestDefaultCostRuleSeeded:
    def test_default_rule_present_after_init(self):
        with get_session() as s:
            rules = s.exec(
                select(CostRuleRow).where(
                    CostRuleRow.provider == DEFAULT_COST_RULE_PROVIDER
                )
            ).all()
        assert len(rules) == 1
        rule = rules[0]
        assert rule.service_type == "llm_chat"
        assert rule.model_name is None
        assert rule.unit_type == "tokens"
        assert rule.input_cost_per_unit_usd == pytest.approx(
            DEFAULT_INPUT_COST_PER_TOKEN_USD
        )
        assert rule.output_cost_per_unit_usd == pytest.approx(
            DEFAULT_OUTPUT_COST_PER_TOKEN_USD
        )

    def test_seed_is_idempotent(self):
        ensure_default_cost_rule()
        ensure_default_cost_rule()
        with get_session() as s:
            count = len(
                s.exec(
                    select(CostRuleRow).where(
                        CostRuleRow.provider == DEFAULT_COST_RULE_PROVIDER
                    )
                ).all()
            )
        assert count == 1

    def test_pricing_matches_request(self):
        # $0.18 / 1M output tokens, $0.039 / 1M input tokens.
        assert DEFAULT_INPUT_COST_PER_TOKEN_USD == pytest.approx(3.9e-8)
        assert DEFAULT_OUTPUT_COST_PER_TOKEN_USD == pytest.approx(1.8e-7)


class TestCostCalculatorFallsBackToDefault:
    def test_unknown_provider_uses_default_rule(self):
        from shadow_loom.cost_calculation import CostCalculator

        user = ensure_example_user()
        with get_session() as s:
            log = AgentCallLogRow(
                user_id=user.id,
                agent_type="Generation",
                agent_name="shadow_loom.generation.SceneAgent",
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                total_tokens=2_000_000,
                model_provider="some-provider-with-no-rule",
                model_name="some-unknown-model",
            )
            s.add(log)
            s.commit()
            s.refresh(log)

            calc = CostCalculator(s)
            cost = calc.calculate_agent_call_cost(log)
        # Raw: 1M input * $0.039/1M + 1M output * $0.18/1M = $0.219
        # × 1.5 default gross-profit multiplier = $0.3285
        assert cost == pytest.approx((0.039 + 0.18) * 1.5)

    def test_provider_specific_rule_still_wins(self):
        from shadow_loom.cost_calculation import CostCalculator

        user = ensure_example_user()
        # Override price for one specific provider.
        with get_session() as s:
            s.add(CostRuleRow(
                provider="openai",
                service_type="llm_chat",
                model_name=None,
                unit_type="tokens",
                input_cost_per_unit_usd=1e-6,
                output_cost_per_unit_usd=2e-6,
            ))
            s.commit()

        with get_session() as s:
            log = AgentCallLogRow(
                user_id=user.id,
                agent_type="Generation",
                agent_name="shadow_loom.generation.SceneAgent",
                prompt_tokens=1000,
                completion_tokens=2000,
                total_tokens=3000,
                model_provider="openai",
                model_name="gpt-4o",
            )
            s.add(log)
            s.commit()
            s.refresh(log)

            calc = CostCalculator(s)
            cost = calc.calculate_agent_call_cost(log)
        # Raw: 1000 * 1e-6 + 2000 * 2e-6 = $0.005
        # × 1.5 default multiplier = $0.0075
        assert cost == pytest.approx(0.005 * 1.5)


class TestLifetimeRollupIncrements:
    def test_example_user_lifetime_counter_grows(self):
        from shadow_loom.cost_calculation import increment_user_lifetime_usage

        user = ensure_example_user()
        with get_session() as s:
            increment_user_lifetime_usage(
                s, user.id,
                agent_tokens=1000, agent_cost_usd=0.05, agent_calls=1,
            )
            increment_user_lifetime_usage(
                s, user.id,
                agent_tokens=500, agent_cost_usd=0.02, agent_calls=1,
            )

        with get_session() as s:
            row = s.exec(
                select(UserUsageSummaryRow).where(
                    UserUsageSummaryRow.user_id == user.id,
                    UserUsageSummaryRow.period_type == "lifetime",
                )
            ).one()
        assert row.total_agent_calls == 2
        assert row.total_agent_tokens == 1500
        assert row.total_agent_cost_usd == pytest.approx(0.07)

    def test_other_user_gets_own_lifetime_row(self):
        from shadow_loom.cost_calculation import increment_user_lifetime_usage

        example = ensure_example_user()
        with get_session() as s:
            other = UserRow(
                provider="local",
                provider_id="local:test_other",
                username="other",
                is_example=False,
            )
            s.add(other)
            s.commit()
            s.refresh(other)
            other_id = other.id

            increment_user_lifetime_usage(
                s, example.id,
                agent_tokens=100, agent_cost_usd=0.01,
            )
            increment_user_lifetime_usage(
                s, other_id,
                agent_tokens=300, agent_cost_usd=0.03,
            )
            increment_user_lifetime_usage(
                s, other_id,
                agent_tokens=200, agent_cost_usd=0.02,
            )

        with get_session() as s:
            rows = s.exec(
                select(UserUsageSummaryRow).where(
                    UserUsageSummaryRow.period_type == "lifetime",
                )
            ).all()
        by_user = {r.user_id: r for r in rows}
        assert by_user[example.id].total_agent_tokens == 100
        assert by_user[example.id].total_agent_cost_usd == pytest.approx(0.01)
        assert by_user[other_id].total_agent_tokens == 500
        assert by_user[other_id].total_agent_cost_usd == pytest.approx(0.05)
        assert by_user[other_id].total_agent_calls == 2

    def test_api_counters_increment_independently(self):
        from shadow_loom.cost_calculation import increment_user_lifetime_usage

        user = ensure_example_user()
        with get_session() as s:
            increment_user_lifetime_usage(
                s, user.id,
                agent_tokens=0, agent_cost_usd=0.0, agent_calls=0,
                api_calls=1, api_cost_usd=0.04,
            )
        with get_session() as s:
            row = s.exec(
                select(UserUsageSummaryRow).where(
                    UserUsageSummaryRow.user_id == user.id,
                )
            ).one()
        assert row.total_api_calls == 1
        assert row.total_api_cost_usd == pytest.approx(0.04)
        assert row.total_agent_calls == 0


class TestGrossProfitMultiplier:
    def test_default_multiplier_is_1_5(self, monkeypatch):
        # Make sure no env override is leaking from the host.
        monkeypatch.delenv("SHADOW_LOOM_COST_GROSS_MARGIN", raising=False)
        from shadow_loom.cost_calculation import (
            DEFAULT_GROSS_PROFIT_MULTIPLIER,
            get_gross_profit_multiplier,
        )
        assert DEFAULT_GROSS_PROFIT_MULTIPLIER == 1.5
        assert get_gross_profit_multiplier() == 1.5

    def test_env_override_applied_to_agent_cost(self, monkeypatch):
        from shadow_loom.cost_calculation import CostCalculator

        monkeypatch.setenv("SHADOW_LOOM_COST_GROSS_MARGIN", "2.0")
        user = ensure_example_user()
        with get_session() as s:
            log = AgentCallLogRow(
                user_id=user.id,
                agent_type="Generation",
                agent_name="shadow_loom.generation.SceneAgent",
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                total_tokens=2_000_000,
                model_provider="some-provider-with-no-rule",
                model_name="some-unknown-model",
            )
            s.add(log)
            s.commit()
            s.refresh(log)

            calc = CostCalculator(s)
            cost = calc.calculate_agent_call_cost(log)
        # Raw $0.219 \u00d7 2.0 override = $0.438
        assert cost == pytest.approx((0.039 + 0.18) * 2.0)

    def test_raw_cost_mode_with_multiplier_1(self, monkeypatch):
        from shadow_loom.cost_calculation import CostCalculator

        monkeypatch.setenv("SHADOW_LOOM_COST_GROSS_MARGIN", "1.0")
        user = ensure_example_user()
        with get_session() as s:
            log = AgentCallLogRow(
                user_id=user.id, agent_type="Generation",
                agent_name="x",
                prompt_tokens=1_000_000, completion_tokens=1_000_000,
                total_tokens=2_000_000,
                model_provider="unknown", model_name=None,
            )
            s.add(log); s.commit(); s.refresh(log)
            cost = CostCalculator(s).calculate_agent_call_cost(log)
        # No margin applied \u2014 raw vendor cost.
        assert cost == pytest.approx(0.039 + 0.18)

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        from shadow_loom.cost_calculation import get_gross_profit_multiplier

        monkeypatch.setenv("SHADOW_LOOM_COST_GROSS_MARGIN", "not-a-number")
        assert get_gross_profit_multiplier() == 1.5

    def test_nonpositive_env_falls_back_to_default(self, monkeypatch):
        from shadow_loom.cost_calculation import get_gross_profit_multiplier

        monkeypatch.setenv("SHADOW_LOOM_COST_GROSS_MARGIN", "-0.5")
        assert get_gross_profit_multiplier() == 1.5
        monkeypatch.setenv("SHADOW_LOOM_COST_GROSS_MARGIN", "0")
        assert get_gross_profit_multiplier() == 1.5

    def test_multiplier_applies_to_api_call_cost(self, monkeypatch):
        from shadow_loom.cost_calculation import CostCalculator

        monkeypatch.setenv("SHADOW_LOOM_COST_GROSS_MARGIN", "1.5")
        user = ensure_example_user()
        with get_session() as s:
            s.add(CostRuleRow(
                provider="tavily", service_type="search", model_name=None,
                unit_type="requests", cost_per_unit_usd=0.01,
            ))
            s.commit()
            log = ApiCallLogRow(
                user_id=user.id, provider="tavily", service_type="search",
            )
            s.add(log); s.commit(); s.refresh(log)
            cost = CostCalculator(s).calculate_api_call_cost(log)
        # 1 request * $0.01 raw * 1.5 = $0.015
        assert cost == pytest.approx(0.015)
