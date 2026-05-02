#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add cost tracking tables to the Shadow-Loom database.

This script adds the new cost tracking infrastructure:
- AgentCallLogRow: Track agent executions
- ApiCallLogRow: Track external API calls
- CostRuleRow: Define pricing rules
- UserUsageSummaryRow: User usage rollups
- ProjectUsageSummaryRow: Project usage rollups

Run with: python scripts/migrate_cost_tracking.py
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from shadow_loom.db import init_db, get_engine, get_session
from shadow_loom.settings import get_settings

logger = logging.getLogger(__name__)


def migrate_cost_tracking():
    """Add the new cost tracking tables and seed with default rules."""
    print("🔧 Migrating Shadow-Loom database for cost tracking...")
    
    # Initialize database with new schema
    settings = get_settings()
    init_db(settings.core.database_url)
    
    print("✅ Cost tracking tables created successfully")
    
    # Insert default cost rules
    session = get_session()
    try:
        # Check if we already have cost rules
        existing_rules = session.exec(text("SELECT COUNT(*) FROM cost_rules")).first()
        if existing_rules and existing_rules[0] > 0:
            print("💡 Cost rules already exist, skipping seed data")
            return
        
        print("🌱 Seeding default cost rules...")
        
        # OpenAI GPT-4o pricing (example - update with current rates)
        session.exec(text("""
            INSERT INTO cost_rules (provider, service_type, model_name, unit_type, 
                                   cost_per_unit_usd, input_cost_per_unit_usd, output_cost_per_unit_usd, description)
            VALUES ('openai', 'llm_chat', 'gpt-4o', 'tokens', 0.000010, 0.000005, 0.000015, 
                    'GPT-4o pricing as of May 2026')
        """))
        
        # OpenAI GPT-4o Mini pricing
        session.exec(text("""
            INSERT INTO cost_rules (provider, service_type, model_name, unit_type,
                                   cost_per_unit_usd, input_cost_per_unit_usd, output_cost_per_unit_usd, description)
            VALUES ('openai', 'llm_chat', 'gpt-4o-mini', 'tokens', 0.000000375, 0.00000015, 0.0000006,
                    'GPT-4o Mini pricing as of May 2026')
        """))
        
        # Claude 3.5 Sonnet pricing (example)
        session.exec(text("""
            INSERT INTO cost_rules (provider, service_type, model_name, unit_type,
                                   cost_per_unit_usd, input_cost_per_unit_usd, output_cost_per_unit_usd, description)
            VALUES ('anthropic', 'llm_chat', 'claude-3-5-sonnet-20241022', 'tokens', 0.000009, 0.000003, 0.000015,
                    'Claude 3.5 Sonnet pricing as of May 2026')
        """))
        
        # Tavily search pricing (example - update with real rates)  
        session.exec(text("""
            INSERT INTO cost_rules (provider, service_type, unit_type, cost_per_unit_usd, description)
            VALUES ('tavily', 'search', 'requests', 0.001, 'Tavily search API per request')
        """))
        
        # Generic fallback for unknown LLM providers
        session.exec(text("""
            INSERT INTO cost_rules (provider, service_type, unit_type, cost_per_unit_usd, description)
            VALUES ('unknown', 'llm_chat', 'tokens', 0.00001, 'Fallback cost estimate for unknown providers')
        """))
        
        session.commit()
        print("✅ Default cost rules inserted")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to seed cost rules: {e}")
        raise
    finally:
        session.close()
    
    print("🎉 Cost tracking migration completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_cost_tracking()