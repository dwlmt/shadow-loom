#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cost tracking management CLI for Shadow-Loom.

This CLI provides basic management commands for the cost tracking system:
- View usage statistics
- Manage cost rules
- Generate reports

Usage:
    python shadow_loom/cost_cli.py --help
    python shadow_loom/cost_cli.py usage --days 7
    python shadow_loom/cost_cli.py rules list
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click
from sqlmodel import select, func

# Add the project root to Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from shadow_loom.db import (
    init_db, get_session, AgentCallLogRow, ApiCallLogRow, CostRuleRow,
    UserRow
)
from shadow_loom.settings import get_settings


@click.group()
@click.option('--db-url', help='Database URL (overrides config)')
def cli(db_url: Optional[str]):
    """Shadow-Loom cost tracking management CLI."""
    settings = get_settings()
    database_url = db_url or settings.core.database_url
    init_db(database_url)


@cli.group()
def usage():
    """View usage statistics and reports."""
    pass


@usage.command('summary')
@click.option('--days', default=30, help='Days to look back')
@click.option('--user-id', type=int, help='Filter by user ID')
@click.option('--project-id', type=int, help='Filter by project ID')
def usage_summary(days: int, user_id: Optional[int], project_id: Optional[int]):
    """Show usage summary for specified period."""
    session = get_session()
    
    try:
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        print(f"📊 Usage Summary ({start_date.date()} to {end_date.date()})")
        print("=" * 60)
        
        # Build query filters
        agent_filters = [
            AgentCallLogRow.created_at >= start_date,
            AgentCallLogRow.created_at <= end_date
        ]
        api_filters = [
            ApiCallLogRow.created_at >= start_date,
            ApiCallLogRow.created_at <= end_date
        ]
        
        if user_id:
            agent_filters.append(AgentCallLogRow.user_id == user_id)
            api_filters.append(ApiCallLogRow.user_id == user_id)
        
        if project_id:
            agent_filters.append(AgentCallLogRow.project_id == project_id)
            api_filters.append(ApiCallLogRow.project_id == project_id)
        
        # Agent call statistics
        agent_stats = session.exec(
            select(
                func.count(AgentCallLogRow.id).label('total_calls'),
                func.sum(AgentCallLogRow.total_tokens).label('total_tokens'),
                func.sum(AgentCallLogRow.estimated_cost_usd).label('total_cost'),
                func.avg(AgentCallLogRow.execution_time_ms).label('avg_time_ms')
            ).where(*agent_filters)
        ).first()
        
        # API call statistics
        api_stats = session.exec(
            select(
                func.count(ApiCallLogRow.id).label('total_calls'),
                func.sum(ApiCallLogRow.estimated_cost_usd).label('total_cost'),
                func.avg(ApiCallLogRow.response_time_ms).label('avg_time_ms')
            ).where(*api_filters)
        ).first()
        
        # Display results
        print("🤖 Agent Calls:")
        print(f"   Total calls: {agent_stats.total_calls or 0:,}")
        print(f"   Total tokens: {agent_stats.total_tokens or 0:,}")
        print(f"   Total cost: ${agent_stats.total_cost or 0:.4f}")
        print(f"   Avg execution time: {agent_stats.avg_time_ms or 0:.1f}ms")
        
        print("\\n🌐 API Calls:")
        print(f"   Total calls: {api_stats.total_calls or 0:,}")
        print(f"   Total cost: ${api_stats.total_cost or 0:.4f}")
        print(f"   Avg response time: {api_stats.avg_time_ms or 0:.1f}ms")
        
        total_cost = (agent_stats.total_cost or 0) + (api_stats.total_cost or 0)
        print(f"\\n💰 Total Cost: ${total_cost:.4f}")
        
        # Top agent types
        print("\\n📈 Top Agent Types:")
        top_agents = session.exec(
            select(
                AgentCallLogRow.agent_type,
                func.count(AgentCallLogRow.id).label('call_count'),
                func.sum(AgentCallLogRow.estimated_cost_usd).label('total_cost')
            ).where(*agent_filters)
            .group_by(AgentCallLogRow.agent_type)
            .order_by(func.count(AgentCallLogRow.id).desc())
            .limit(5)
        ).all()
        
        for agent in top_agents:
            print(f"   {agent.agent_type}: {agent.call_count} calls, ${agent.total_cost or 0:.4f}")
        
    finally:
        session.close()


@usage.command('by-user')
@click.option('--days', default=30, help='Days to look back')
@click.option('--top', default=10, help='Number of top users to show')
def usage_by_user(days: int, top: int):
    """Show usage breakdown by user."""
    session = get_session()
    
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        print(f"👥 Top {top} Users by Usage ({start_date.date()} to {end_date.date()})")
        print("=" * 80)
        
        # Query user usage with joins
        user_usage = session.exec(
            select(
                UserRow.username,
                UserRow.display_name,
                func.count(AgentCallLogRow.id).label('agent_calls'),
                func.sum(AgentCallLogRow.total_tokens).label('total_tokens'),
                func.sum(AgentCallLogRow.estimated_cost_usd).label('agent_cost'),
            ).select_from(UserRow)
            .join(AgentCallLogRow, UserRow.id == AgentCallLogRow.user_id)
            .where(
                AgentCallLogRow.created_at >= start_date,
                AgentCallLogRow.created_at <= end_date
            )
            .group_by(UserRow.id, UserRow.username, UserRow.display_name)
            .order_by(func.sum(AgentCallLogRow.estimated_cost_usd).desc())
            .limit(top)
        ).all()
        
        for user in user_usage:
            name = user.display_name or user.username
            print(f"   {name}:")
            print(f"      Agent calls: {user.agent_calls:,}")
            print(f"      Tokens used: {user.total_tokens or 0:,}")
            print(f"      Cost: ${user.agent_cost or 0:.4f}")
            print()
        
    finally:
        session.close()


@cli.group()
def rules():
    """Manage cost rules."""
    pass


@rules.command('list')
@click.option('--provider', help='Filter by provider')
@click.option('--service-type', help='Filter by service type')
def rules_list(provider: Optional[str], service_type: Optional[str]):
    """List cost rules."""
    session = get_session()
    
    try:
        query = select(CostRuleRow)
        
        if provider:
            query = query.where(CostRuleRow.provider == provider)
        if service_type:
            query = query.where(CostRuleRow.service_type == service_type)
        
        query = query.order_by(CostRuleRow.provider, CostRuleRow.service_type)
        
        rules = session.exec(query).all()
        
        print("💰 Cost Rules")
        print("=" * 80)
        
        for rule in rules:
            print(f"Provider: {rule.provider}")
            print(f"Service: {rule.service_type}")
            if rule.model_name:
                print(f"Model: {rule.model_name}")
            print(f"Unit Type: {rule.unit_type}")
            
            if rule.input_cost_per_unit_usd and rule.output_cost_per_unit_usd:
                print(f"Cost: ${rule.input_cost_per_unit_usd:.6f} input, ${rule.output_cost_per_unit_usd:.6f} output")
            else:
                print(f"Cost: ${rule.cost_per_unit_usd:.6f} per {rule.unit_type}")
            
            if rule.description:
                print(f"Description: {rule.description}")
            print(f"Effective from: {rule.effective_from}")
            print("-" * 40)
        
        print(f"\\nTotal rules: {len(rules)}")
        
    finally:
        session.close()


@rules.command('add')
@click.option('--provider', required=True, help='Provider name (e.g., openai)')
@click.option('--service-type', required=True, help='Service type (e.g., llm_chat)')
@click.option('--model-name', help='Model name (optional)')
@click.option('--unit-type', required=True, help='Unit type (tokens, requests, etc.)')
@click.option('--cost-per-unit', type=float, required=True, help='Base cost per unit')
@click.option('--input-cost', type=float, help='Input cost per unit (for LLMs)')
@click.option('--output-cost', type=float, help='Output cost per unit (for LLMs)')
@click.option('--description', help='Rule description')
def rules_add(provider: str, service_type: str, model_name: Optional[str], 
              unit_type: str, cost_per_unit: float, input_cost: Optional[float],
              output_cost: Optional[float], description: Optional[str]):
    """Add a new cost rule."""
    session = get_session()
    
    try:
        rule = CostRuleRow(
            provider=provider,
            service_type=service_type,
            model_name=model_name,
            unit_type=unit_type,
            cost_per_unit_usd=cost_per_unit,
            input_cost_per_unit_usd=input_cost,
            output_cost_per_unit_usd=output_cost,
            description=description
        )
        
        session.add(rule)
        session.commit()
        
        print(f"✅ Added cost rule for {provider}:{service_type}")
        if model_name:
            print(f"   Model: {model_name}")
        print(f"   Cost: ${cost_per_unit:.6f} per {unit_type}")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Failed to add cost rule: {e}")
        raise
    finally:
        session.close()


if __name__ == '__main__':
    cli()