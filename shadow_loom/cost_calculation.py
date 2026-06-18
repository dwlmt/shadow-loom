# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cost calculation engine for Shadow-Loom.

This module provides cost calculation services that:
1. Apply cost rules to logged agent and API calls
2. Update cost estimates in the database
3. Generate usage summary rollups
4. Provide background job functionality
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select, text

from shadow_loom.db import (
    get_session, AgentCallLogRow, ApiCallLogRow, CostRuleRow,
    UserUsageSummaryRow, ProjectUsageSummaryRow,
    DEFAULT_COST_RULE_PROVIDER,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Gross-profit multiplier
# =====================================================================

# Applied to every raw token / request cost computed from a
# ``CostRuleRow`` before the value is returned to the caller (and so
# before it is persisted on ``estimated_cost_usd`` and folded into the
# per-user lifetime rollup). Lets operators bill above raw vendor
# token cost without rewriting every cost rule. The default of 1.5
# yields a 50 % gross margin on top of the raw OpenAI / Anthropic /
# Ollama unit price; override via the ``SHADOW_LOOM_COST_GROSS_MARGIN``
# env var (any positive float; e.g. ``2.0`` for a 100 % markup, or
# ``1.0`` to bill at raw vendor cost with no margin).
DEFAULT_GROSS_PROFIT_MULTIPLIER: float = 1.5


def get_gross_profit_multiplier() -> float:
    """Return the active gross-profit multiplier.

    Reads ``SHADOW_LOOM_COST_GROSS_MARGIN`` from the environment at
    call time so operators can flip the value without restarting; a
    malformed or non-positive value logs a warning and falls back to
    :data:`DEFAULT_GROSS_PROFIT_MULTIPLIER`.
    """
    raw = os.environ.get("SHADOW_LOOM_COST_GROSS_MARGIN")
    if raw is None or raw == "":
        return DEFAULT_GROSS_PROFIT_MULTIPLIER
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid SHADOW_LOOM_COST_GROSS_MARGIN=%r; using default %.2f.",
            raw, DEFAULT_GROSS_PROFIT_MULTIPLIER,
        )
        return DEFAULT_GROSS_PROFIT_MULTIPLIER
    if value <= 0.0:
        logger.warning(
            "SHADOW_LOOM_COST_GROSS_MARGIN=%g must be > 0; using default %.2f.",
            value, DEFAULT_GROSS_PROFIT_MULTIPLIER,
        )
        return DEFAULT_GROSS_PROFIT_MULTIPLIER
    return value


# =====================================================================
# Lifetime usage increment (per-call hook)
# =====================================================================

# Sentinel ``period_start`` used for the lifetime rollup row. The
# schema's ``UniqueConstraint("user_id", "period_start")`` requires a
# concrete value, so we use the Unix epoch as the lifetime sentinel:
# every (user_id, EPOCH) pair is unique and clearly distinct from any
# real daily/monthly period.
_LIFETIME_PERIOD_START = datetime(1970, 1, 1, tzinfo=timezone.utc)


def increment_user_lifetime_usage(
    session: Session,
    user_id: int,
    *,
    agent_tokens: int = 0,
    agent_cost_usd: float = 0.0,
    agent_calls: int = 1,
    api_calls: int = 0,
    api_cost_usd: float = 0.0,
) -> UserUsageSummaryRow:
    """Increment the lifetime ``UserUsageSummaryRow`` for ``user_id``.

    Called from :mod:`shadow_loom._agent_logging` immediately after
    each agent call's ``estimated_cost_usd`` is stamped so the
    per-user running tally moves in lockstep with the per-row log.
    Without this hook the only path to a non-zero summary row was
    the nightly :meth:`UsageSummaryCalculator.update_user_summaries`
    batch \u2014 anything between batch runs was invisible to the UI
    cost panel.

    Idempotent in the row-creation sense (re-running creates only
    the first lifetime row); the *counters* increment monotonically
    on every call by design.
    """
    def _lifetime_row() -> Optional[UserUsageSummaryRow]:
        return session.exec(
            select(UserUsageSummaryRow).where(
                UserUsageSummaryRow.user_id == user_id,
                UserUsageSummaryRow.period_type == "lifetime",
                UserUsageSummaryRow.period_start == _LIFETIME_PERIOD_START,
            )
        ).first()

    def _apply_deltas(r: UserUsageSummaryRow) -> None:
        r.total_agent_calls = int(r.total_agent_calls or 0) + int(agent_calls)
        r.total_agent_tokens = int(r.total_agent_tokens or 0) + int(agent_tokens)
        r.total_agent_cost_usd = float(r.total_agent_cost_usd or 0.0) + float(agent_cost_usd)
        r.total_api_calls = int(r.total_api_calls or 0) + int(api_calls)
        r.total_api_cost_usd = float(r.total_api_cost_usd or 0.0) + float(api_cost_usd)
        r.updated_at = datetime.now(timezone.utc)

    row = _lifetime_row()
    if row is None:
        row = UserUsageSummaryRow(
            user_id=user_id,
            period_type="lifetime",
            period_start=_LIFETIME_PERIOD_START,
            period_end=None,
            total_agent_calls=int(agent_calls),
            total_agent_tokens=int(agent_tokens),
            total_agent_cost_usd=float(agent_cost_usd),
            total_api_calls=int(api_calls),
            total_api_cost_usd=float(api_cost_usd),
        )
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            # A concurrent first-call inserted the lifetime row between
            # our SELECT and INSERT (uq_user_period). Fold our increment
            # into the existing row instead of silently dropping it.
            session.rollback()
            row = _lifetime_row()
            if row is None:
                raise
            _apply_deltas(row)
            session.commit()
    else:
        _apply_deltas(row)
        session.commit()
    session.refresh(row)
    return row


class CostCalculator:
    """Calculate costs for agent and API calls based on usage rules."""
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()
        self._cost_rules_cache: Dict[str, CostRuleRow] = {}
        self._cache_expiry = datetime.now(timezone.utc)
        
    def _refresh_cost_rules_cache(self) -> None:
        """Refresh the cost rules cache if expired."""
        now = datetime.now(timezone.utc)
        if now > self._cache_expiry:
            self._cost_rules_cache.clear()
            self._cache_expiry = now + timedelta(minutes=15)  # Cache for 15 minutes
        
    def get_cost_rule(
        self, 
        provider: str, 
        service_type: str, 
        model_name: Optional[str] = None
    ) -> Optional[CostRuleRow]:
        """Get applicable cost rule for a service."""
        self._refresh_cost_rules_cache()

        cache_key = f"{provider}:{service_type}:{model_name or ''}"

        if cache_key in self._cost_rules_cache:
            return self._cost_rules_cache[cache_key]

        # Try to find exact model match first
        query = select(CostRuleRow).where(
            CostRuleRow.provider == provider,
            CostRuleRow.service_type == service_type,
            (CostRuleRow.effective_from.is_(None) |
             (CostRuleRow.effective_from <= datetime.now(timezone.utc))),
            (CostRuleRow.effective_to.is_(None) |
             (CostRuleRow.effective_to > datetime.now(timezone.utc)))
        ).order_by(CostRuleRow.created_at.desc())

        if model_name:
            model_query = query.where(CostRuleRow.model_name == model_name)
            rule = self.session.exec(model_query).first()
            if rule:
                self._cost_rules_cache[cache_key] = rule
                return rule

        # Fall back to generic rule for provider/service
        generic_query = query.where(CostRuleRow.model_name.is_(None))
        rule = self.session.exec(generic_query).first()
        if rule is None and provider != DEFAULT_COST_RULE_PROVIDER:
            # Final fallback: the catch-all ``provider="default"``
            # row seeded by ``ensure_default_cost_rule()`` on
            # ``init_db()``. Without this every (provider, model)
            # pair without an explicit cost rule was priced at $0
            # \u2014 the per-user / per-project rollups in
            # ``UserUsageSummaryRow`` then summed to zero too,
            # which is exactly the "no costs ever logged" symptom
            # the hookup task fixed.
            default_query = select(CostRuleRow).where(
                CostRuleRow.provider == DEFAULT_COST_RULE_PROVIDER,
                CostRuleRow.service_type == service_type,
                CostRuleRow.model_name.is_(None),
            ).order_by(CostRuleRow.created_at.desc())
            rule = self.session.exec(default_query).first()
        # Only cache positive hits: caching a None would mask a cost rule
        # seeded after this (provider, model) was first queried, pricing
        # calls at $0 until the 15-min TTL lapses.
        if rule is not None:
            self._cost_rules_cache[cache_key] = rule
        return rule
    
    def calculate_agent_call_cost(self, log_entry: AgentCallLogRow) -> float:
        """Calculate cost for an agent call based on token usage."""
        if not log_entry.total_tokens:
            return 0.0
            
        # Use model provider/name if available, otherwise default to 'unknown'
        provider = log_entry.model_provider or "unknown"
        model_name = log_entry.model_name
        
        rule = self.get_cost_rule(
            provider=provider,
            service_type="llm_chat",
            model_name=model_name
        )
        
        if not rule:
            logger.warning(
                f"No cost rule found for provider={provider}, model={model_name}. "
                f"Agent call {log_entry.id} will have $0 cost."
            )
            return 0.0
            
        # Apply pricing based on rule structure
        # Use ``is not None`` so legitimately-zero pricing (free models /
        # introductory tiers) still routes through the split-pricing
        # branch. With truthiness the 0.0 falls through to the combined-
        # token branch which mis-prices anything where one side is
        # explicitly free.
        if (
            rule.input_cost_per_unit_usd is not None
            and rule.output_cost_per_unit_usd is not None
        ):
            # Separate input/output pricing (preferred for LLMs)
            input_tokens = log_entry.prompt_tokens or 0
            output_tokens = log_entry.completion_tokens or 0
            
            input_cost = input_tokens * rule.input_cost_per_unit_usd
            output_cost = output_tokens * rule.output_cost_per_unit_usd
            raw_total = input_cost + output_cost
            multiplier = get_gross_profit_multiplier()
            total_cost = raw_total * multiplier
            
            logger.debug(
                f"Calculated agent call cost: {input_tokens} input tokens * "
                f"${rule.input_cost_per_unit_usd:.6f} + {output_tokens} output tokens * "
                f"${rule.output_cost_per_unit_usd:.6f} = ${raw_total:.6f} raw "
                f"× {multiplier:.3f} margin = ${total_cost:.6f}"
            )
            return total_cost
        else:
            # Combined token pricing
            raw_total = log_entry.total_tokens * rule.cost_per_unit_usd
            multiplier = get_gross_profit_multiplier()
            total_cost = raw_total * multiplier
            logger.debug(
                f"Calculated agent call cost: {log_entry.total_tokens} tokens * "
                f"${rule.cost_per_unit_usd:.6f} = ${raw_total:.6f} raw "
                f"× {multiplier:.3f} margin = ${total_cost:.6f}"
            )
            return total_cost
            
    def calculate_api_call_cost(self, log_entry: ApiCallLogRow) -> float:
        """Calculate cost for an external API call.""" 
        rule = self.get_cost_rule(
            provider=log_entry.provider,
            service_type=log_entry.service_type
        )
        
        if not rule:
            logger.warning(
                f"No cost rule found for provider={log_entry.provider}, "
                f"service_type={log_entry.service_type}. "
                f"API call {log_entry.id} will have $0 cost."
            )
            return 0.0

        # Round-11 R11-05: do not bill the user for failed external
        # API calls. Provider-side 4xx/5xx responses either return no
        # payload or an error envelope, but the per-request /
        # per-result pricing branches below would still charge full
        # price for the request. Treat any non-2xx (3xx redirects
        # included — we did not get the data we asked for) as $0 so
        # cost dashboards reflect billable work only.
        status_code = getattr(log_entry, "status_code", None)
        if status_code is not None and not (200 <= status_code < 300):
            logger.debug(
                "Skipping cost for API call %s: status=%s (non-2xx, not billed)",
                getattr(log_entry, "id", "?"), status_code,
            )
            return 0.0

        # Apply pricing based on unit type
        if rule.unit_type == "requests":
            raw_cost = rule.cost_per_unit_usd
        elif rule.unit_type == "results" and log_entry.results_count:
            raw_cost = log_entry.results_count * rule.cost_per_unit_usd
        elif rule.unit_type == "tokens" and log_entry.request_size:
            raw_cost = log_entry.request_size * rule.cost_per_unit_usd
        elif rule.unit_type == "characters" and log_entry.request_size:
            raw_cost = log_entry.request_size * rule.cost_per_unit_usd
        else:
            # Default to per-request pricing
            raw_cost = rule.cost_per_unit_usd

        multiplier = get_gross_profit_multiplier()
        cost = raw_cost * multiplier

        logger.debug(
            f"Calculated API call cost: {rule.unit_type} * "
            f"${rule.cost_per_unit_usd:.6f} = ${raw_cost:.6f} raw "
            f"× {multiplier:.3f} margin = ${cost:.6f}"
        )
        return cost
        
    def update_costs_batch(self, limit: int = 1000) -> Tuple[int, int]:
        """Update costs for entries that haven't been calculated yet.

        Returns:
            Tuple of (agent_entries_updated, api_entries_updated)

        Note on failed calls: rows with ``status != "success"`` are also
        processed. Most providers (Ollama, OpenAI, Anthropic) do not bill
        for fully-failed requests, but partial responses with usable
        ``prompt_tokens`` / ``completion_tokens`` *are* billable. The
        per-row calculator already returns ``0.0`` when ``total_tokens``
        is falsy, so this naturally bills the partial-response case and
        zeroes the all-failed case. Stamping all rows here also prevents
        the "is None" filter from re-scanning failed calls forever.
        """
        agent_count = 0
        api_count = 0

        try:
            # Update agent call costs (all statuses; see docstring).
            agent_query = select(AgentCallLogRow).where(
                AgentCallLogRow.estimated_cost_usd.is_(None),
            ).limit(limit)

            agent_entries = self.session.exec(agent_query).all()

            for entry in agent_entries:
                try:
                    entry.estimated_cost_usd = self.calculate_agent_call_cost(entry)
                    agent_count += 1
                except Exception as e:
                    logger.error(f"Failed to calculate cost for agent call {entry.id}: {e}")
                    entry.estimated_cost_usd = 0.0

            # Update API call costs (all statuses; see docstring).
            api_query = select(ApiCallLogRow).where(
                ApiCallLogRow.estimated_cost_usd.is_(None),
            ).limit(limit)
            
            api_entries = self.session.exec(api_query).all()
            
            for entry in api_entries:
                try:
                    entry.estimated_cost_usd = self.calculate_api_call_cost(entry)
                    api_count += 1
                except Exception as e:
                    logger.error(f"Failed to calculate cost for API call {entry.id}: {e}")
                    entry.estimated_cost_usd = 0.0
            
            self.session.commit()
            logger.info(f"Updated costs for {agent_count} agent calls and {api_count} API calls")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to update costs batch: {e}")
            raise
            
        return agent_count, api_count


class UsageSummaryCalculator:
    """Calculate and maintain usage summary rollups."""
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()
    
    def update_user_summaries(
        self, 
        period_type: str = "daily",
        target_date: Optional[datetime] = None
    ) -> int:
        """Update user usage summaries for the specified period.
        
        Args:
            period_type: "daily" or "monthly"
            target_date: Date to calculate for (defaults to yesterday/last month)
            
        Returns:
            Number of user summaries updated
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc)
            
        if period_type == "daily":
            # Calculate for yesterday to ensure complete data
            target_date = target_date - timedelta(days=1)
            period_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
        elif period_type == "monthly":
            # Calculate for last month
            if target_date.month == 1:
                period_start = target_date.replace(year=target_date.year - 1, month=12, day=1,
                                                 hour=0, minute=0, second=0, microsecond=0)
            else:
                period_start = target_date.replace(month=target_date.month - 1, day=1,
                                                 hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate end of month
            if period_start.month == 12:
                period_end = period_start.replace(year=period_start.year + 1, month=1, day=1)
            else:
                period_end = period_start.replace(month=period_start.month + 1, day=1)
        else:
            raise ValueError(f"Unsupported period_type: {period_type}")
        
        logger.info(f"Updating {period_type} user summaries for {period_start.date()} to {period_end.date()}")
        
        # Query user stats with raw SQL for better performance
        user_stats_query = text("""
            SELECT 
                u.id as user_id,
                COALESCE(agent_stats.call_count, 0) as agent_calls,
                COALESCE(agent_stats.token_count, 0) as agent_tokens,
                COALESCE(agent_stats.agent_cost, 0) as agent_cost,
                COALESCE(api_stats.call_count, 0) as api_calls,
                COALESCE(api_stats.api_cost, 0) as api_cost
            FROM users u
            LEFT JOIN (
                SELECT 
                    user_id,
                    COUNT(*) as call_count,
                    SUM(total_tokens) as token_count,
                    SUM(COALESCE(estimated_cost_usd, 0)) as agent_cost
                FROM agent_call_logs 
                WHERE created_at >= :period_start AND created_at < :period_end
                GROUP BY user_id
            ) agent_stats ON u.id = agent_stats.user_id
            LEFT JOIN (
                SELECT 
                    user_id,
                    COUNT(*) as call_count,
                    SUM(COALESCE(estimated_cost_usd, 0)) as api_cost
                FROM api_call_logs
                WHERE created_at >= :period_start AND created_at < :period_end  
                GROUP BY user_id
            ) api_stats ON u.id = api_stats.user_id
            WHERE agent_stats.call_count > 0 OR api_stats.call_count > 0
        """)
        
        results = self.session.exec(user_stats_query, params={
            "period_start": period_start,
            "period_end": period_end
        }).all()
        
        updated_count = 0
        
        for row in results:
            try:
                # Check if summary already exists
                existing = self.session.exec(
                    select(UserUsageSummaryRow).where(
                        UserUsageSummaryRow.user_id == row.user_id,
                        UserUsageSummaryRow.period_start == period_start,
                        UserUsageSummaryRow.period_type == period_type
                    )
                ).first()
                
                if existing:
                    # Update existing summary
                    existing.total_agent_calls = row.agent_calls
                    existing.total_agent_tokens = row.agent_tokens  
                    existing.total_agent_cost_usd = float(row.agent_cost)
                    existing.total_api_calls = row.api_calls
                    existing.total_api_cost_usd = float(row.api_cost)
                    existing.period_end = period_end
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new summary
                    summary = UserUsageSummaryRow(
                        user_id=row.user_id,
                        period_type=period_type,
                        period_start=period_start,
                        period_end=period_end,
                        total_agent_calls=row.agent_calls,
                        total_agent_tokens=row.agent_tokens,
                        total_agent_cost_usd=float(row.agent_cost), 
                        total_api_calls=row.api_calls,
                        total_api_cost_usd=float(row.api_cost)
                    )
                    self.session.add(summary)
                
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update user summary for user {row.user_id}: {e}")
        
        try:
            self.session.commit()
            logger.info(f"Updated {updated_count} user summaries for {period_type} period")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to commit user summaries: {e}")
            raise
            
        return updated_count
    
    def update_project_summaries(
        self, 
        period_type: str = "daily",
        target_date: Optional[datetime] = None
    ) -> int:
        """Update project usage summaries for the specified period."""
        if target_date is None:
            target_date = datetime.now(timezone.utc)
            
        if period_type == "daily":
            target_date = target_date - timedelta(days=1)
            period_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
        elif period_type == "monthly":
            if target_date.month == 1:
                period_start = target_date.replace(year=target_date.year - 1, month=12, day=1,
                                                 hour=0, minute=0, second=0, microsecond=0)
            else:
                period_start = target_date.replace(month=target_date.month - 1, day=1,
                                                 hour=0, minute=0, second=0, microsecond=0)
            
            if period_start.month == 12:
                period_end = period_start.replace(year=period_start.year + 1, month=1, day=1)
            else:
                period_end = period_start.replace(month=period_start.month + 1, day=1)
        else:
            raise ValueError(f"Unsupported period_type: {period_type}")
        
        logger.info(f"Updating {period_type} project summaries for {period_start.date()} to {period_end.date()}")
        
        # Query project stats
        project_stats_query = text("""
            SELECT 
                p.id as project_id,
                COALESCE(agent_stats.call_count, 0) as agent_calls,
                COALESCE(agent_stats.token_count, 0) as agent_tokens,
                COALESCE(agent_stats.agent_cost, 0) as agent_cost,
                COALESCE(api_stats.call_count, 0) as api_calls,
                COALESCE(api_stats.api_cost, 0) as api_cost,
                COALESCE(version_stats.version_count, 0) as versions_created
            FROM projects p
            LEFT JOIN (
                SELECT 
                    project_id,
                    COUNT(*) as call_count,
                    SUM(total_tokens) as token_count,
                    SUM(COALESCE(estimated_cost_usd, 0)) as agent_cost
                FROM agent_call_logs 
                WHERE created_at >= :period_start AND created_at < :period_end
                  AND project_id IS NOT NULL
                GROUP BY project_id
            ) agent_stats ON p.id = agent_stats.project_id
            LEFT JOIN (
                SELECT 
                    project_id,
                    COUNT(*) as call_count,
                    SUM(COALESCE(estimated_cost_usd, 0)) as api_cost
                FROM api_call_logs
                WHERE created_at >= :period_start AND created_at < :period_end
                  AND project_id IS NOT NULL
                GROUP BY project_id
            ) api_stats ON p.id = api_stats.project_id
            LEFT JOIN (
                SELECT 
                    project_id,
                    COUNT(*) as version_count
                FROM versions
                WHERE created_at >= :period_start AND created_at < :period_end
                GROUP BY project_id
            ) version_stats ON p.id = version_stats.project_id
            WHERE agent_stats.call_count > 0 OR api_stats.call_count > 0 OR version_stats.version_count > 0
        """)
        
        results = self.session.exec(project_stats_query, params={
            "period_start": period_start,
            "period_end": period_end
        }).all()
        
        updated_count = 0
        
        for row in results:
            try:
                existing = self.session.exec(
                    select(ProjectUsageSummaryRow).where(
                        ProjectUsageSummaryRow.project_id == row.project_id,
                        ProjectUsageSummaryRow.period_start == period_start,
                        ProjectUsageSummaryRow.period_type == period_type
                    )
                ).first()
                
                if existing:
                    existing.total_agent_calls = row.agent_calls
                    existing.total_agent_tokens = row.agent_tokens
                    existing.total_agent_cost_usd = float(row.agent_cost)
                    existing.total_api_calls = row.api_calls
                    existing.total_api_cost_usd = float(row.api_cost)
                    existing.versions_created = row.versions_created
                    existing.period_end = period_end
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    summary = ProjectUsageSummaryRow(
                        project_id=row.project_id,
                        period_type=period_type,
                        period_start=period_start,
                        period_end=period_end,
                        total_agent_calls=row.agent_calls,
                        total_agent_tokens=row.agent_tokens,
                        total_agent_cost_usd=float(row.agent_cost),
                        total_api_calls=row.api_calls,
                        total_api_cost_usd=float(row.api_cost),
                        versions_created=row.versions_created
                    )
                    self.session.add(summary)
                
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update project summary for project {row.project_id}: {e}")
        
        try:
            self.session.commit()
            logger.info(f"Updated {updated_count} project summaries for {period_type} period")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to commit project summaries: {e}")
            raise
            
        return updated_count


def run_cost_update_job(limit: int = 5000) -> Dict[str, int]:
    """Run a complete cost update job.
    
    This function:
    1. Updates cost estimates for unprocessed entries
    2. Updates daily summaries for yesterday
    3. Updates monthly summaries if it's the first of the month
    
    Returns:
        Dictionary with job statistics
    """
    logger.info("Starting cost update job...")
    stats = {
        "agent_costs_updated": 0,
        "api_costs_updated": 0,
        "user_summaries_updated": 0,
        "project_summaries_updated": 0
    }
    
    session = get_session()
    try:
        # Phase 1: Update individual cost estimates
        calculator = CostCalculator(session)
        agent_count, api_count = calculator.update_costs_batch(limit)
        stats["agent_costs_updated"] = agent_count
        stats["api_costs_updated"] = api_count
        
        # Phase 2: Update usage summaries
        summary_calculator = UsageSummaryCalculator(session)
        
        # Always update daily summaries for yesterday
        user_daily = summary_calculator.update_user_summaries("daily")
        project_daily = summary_calculator.update_project_summaries("daily")
        stats["user_summaries_updated"] += user_daily
        stats["project_summaries_updated"] += project_daily
        
        # Update monthly summaries on the first few days of the month
        now = datetime.now(timezone.utc)
        if now.day <= 3:  # Update monthly summaries in first 3 days
            try:
                user_monthly = summary_calculator.update_user_summaries("monthly")
                project_monthly = summary_calculator.update_project_summaries("monthly") 
                stats["user_summaries_updated"] += user_monthly
                stats["project_summaries_updated"] += project_monthly
                logger.info("Updated monthly summaries")
            except Exception as e:
                logger.error(f"Failed to update monthly summaries: {e}")
        
        logger.info(f"Cost update job completed: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Cost update job failed: {e}")
        raise
    finally:
        session.close()