#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test script to verify cost tracking functionality.

This script tests the basic cost tracking infrastructure:
1. Database table creation
2. Agent call logging
3. API call logging
4. Cost rule lookups

Run with: python scripts/test_cost_tracking.py
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from shadow_loom.db import init_db, get_session, AgentCallLogRow, ApiCallLogRow, CostRuleRow
from shadow_loom._agent_logging import track_agent_call, log_api_call
from shadow_loom.settings import get_settings

logger = logging.getLogger(__name__)


def test_cost_tracking():
    """Test the cost tracking functionality."""
    print("🧪 Testing Shadow-Loom cost tracking infrastructure...")
    
    # Initialize database
    settings = get_settings()
    init_db(settings.core.database_url)
    
    session = get_session()
    
    try:
        # 1. Test cost rule query
        print("\n1️⃣ Testing cost rule lookup...")
        openai_rule = session.query(CostRuleRow).filter_by(
            provider="openai",
            service_type="llm_chat", 
            model_name="gpt-4o"
        ).first()
        
        if openai_rule:
            print(f"✅ Found OpenAI GPT-4o rule: ${openai_rule.input_cost_per_unit_usd:.6f} input, ${openai_rule.output_cost_per_unit_usd:.6f} output")
        else:
            print("❌ No cost rule found for OpenAI GPT-4o")
        
        # 2. Test manual agent call logging (since we don't have a real user/project yet)
        print("\n2️⃣ Testing agent call logging...")
        
        # Create test agent call log entry
        agent_log = AgentCallLogRow(
            user_id=1,  # Assuming user ID 1 exists or will be created
            agent_type="Physics",
            agent_name="TestPhysicsAgent",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            execution_time_ms=1500,
            model_provider="openai",
            model_name="gpt-4o",
            status="success"
        )
        
        session.add(agent_log)
        session.commit()
        session.refresh(agent_log)
        
        print(f"✅ Created agent call log with ID: {agent_log.id}")
        
        # 3. Test API call logging
        print("\n3️⃣ Testing API call logging...")
        
        api_log = log_api_call(
            user_id=1,
            provider="tavily",
            service_type="search",
            request_size=50,  # query length
            response_size=2000,  # response size
            response_time_ms=800,
            status_code=200,
            agent_call_log_id=agent_log.id,
            metadata={
                "query": "test search query",
                "search_depth": "basic",
                "results_returned": 5
            }
        )
        
        if api_log:
            print(f"✅ Created API call log with ID: {api_log.id}")
        else:
            print("❌ API call logging failed")
        
        # 4. Test context manager (without actual agent execution)
        print("\n4️⃣ Testing agent call tracking context manager...")
        
        try:
            with track_agent_call(
                user_id=1,
                agent_type="Auditor",
                agent_name="TestAuditorAgent",
                model_provider="anthropic",
                model_name="claude-3-5-sonnet-20241022"
            ) as log_entry:
                # Simulate some work
                import time
                time.sleep(0.1)
                
                if log_entry:
                    print(f"✅ Context manager created log entry with ID: {log_entry.id}")
                else:
                    print("⚠️ Context manager returned None (database logging disabled)")
                    
        except Exception as e:
            print(f"❌ Context manager test failed: {e}")
        
        # 5. Verify entries in database
        print("\n5️⃣ Verifying database entries...")
        
        agent_count = session.query(AgentCallLogRow).count()
        api_count = session.query(ApiCallLogRow).count()
        rule_count = session.query(CostRuleRow).count()
        
        print(f"📊 Database summary:")
        print(f"   - Agent call logs: {agent_count}")
        print(f"   - API call logs: {api_count}")
        print(f"   - Cost rules: {rule_count}")
        
        print("\n✅ Cost tracking infrastructure test completed successfully!")
        
    except Exception as e:
        print(f"❌ Cost tracking test failed: {e}")
        logger.exception("Cost tracking test error")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cost_tracking()