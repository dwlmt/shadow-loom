# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Usage & Cost Dashboard components for Shadow Loom UI."""

from nicegui import ui

from shadow_loom_ui import db
from shadow_loom_ui.state import AppState
from shadow_loom_ui.theme import feather

# CSS classes from dashboard.py for consistency
SECTION_TITLE_CLS = "text-xl font-semibold text-slate-800 mb-4"
CARD_CLS = "bg-white/80 backdrop-blur-sm border border-slate-200 rounded-xl"


def render_usage_dashboard(state: AppState) -> None:
    """Render usage and cost dashboard section."""
    ui.label("Usage & Costs").classes(SECTION_TITLE_CLS + " mt-6")
    
    # Get user's lifetime usage
    lifetime_usage = db.get_user_lifetime_usage(state.user_id)
    if not lifetime_usage:
        with ui.card().classes(CARD_CLS + " p-6 text-center"):
            ui.label("No usage data yet").classes("text-slate-500")
            ui.label("Your usage will appear here after running agents or research.").classes("text-xs text-slate-400 mt-1")
        return
        
    # Overview metrics row
    with ui.row().classes("w-full gap-4 mb-6"):
        # Total cost card
        with ui.card().classes("bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 p-6"):
            with ui.row().classes("items-center gap-3"):
                feather("dollar-sign", color="#3B82F6", size="lg")
                with ui.column().classes("gap-0"):
                    ui.label(f"${lifetime_usage['total_cost_usd']:.4f}").classes(
                        "text-2xl font-bold text-blue-800"
                    )
                    ui.label("Total Cost (Lifetime)").classes(
                        "text-sm text-blue-600"
                    )
        
        # Agent calls card
        with ui.card().classes("bg-gradient-to-br from-green-50 to-green-100 border-green-200 p-6"):
            with ui.row().classes("items-center gap-3"):
                feather("cpu", color="#10B981", size="lg")
                with ui.column().classes("gap-0"):
                    ui.label(f"{lifetime_usage['total_agent_calls']:,}").classes(
                        "text-2xl font-bold text-green-800"
                    )
                    ui.label("Agent Calls").classes(
                        "text-sm text-green-600"
                    )
        
        # Token usage card
        with ui.card().classes("bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 p-6"):
            with ui.row().classes("items-center gap-3"):
                feather("hash", color="#8B5CF6", size="lg")
                with ui.column().classes("gap-0"):
                    ui.label(f"{lifetime_usage['total_agent_tokens']:,}").classes(
                        "text-2xl font-bold text-purple-800"
                    )
                    ui.label("Tokens Used").classes(
                        "text-sm text-purple-600"
                    )
        
        # API calls card
        with ui.card().classes("bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200 p-6"):
            with ui.row().classes("items-center gap-3"):
                feather("globe", color="#F97316", size="lg")
                with ui.column().classes("gap-0"):
                    ui.label(f"{lifetime_usage['total_api_calls']:,}").classes(
                        "text-2xl font-bold text-orange-800"
                    )
                    ui.label("API Calls").classes(
                        "text-sm text-orange-600"
                    )
    
    # Detailed breakdown in two columns
    with ui.row().classes("w-full gap-6"):
        # Project usage breakdown (left column)
        with ui.column().classes("flex-1"):
            ui.label("Usage by Project").classes("text-lg font-semibold text-slate-700 mb-3")
            project_summaries = db.get_project_usage_summaries(state.user_id, "monthly", 8)
            
            if project_summaries:
                with ui.card().classes(CARD_CLS + " p-0"):
                    for i, proj in enumerate(project_summaries):
                        with ui.row().classes(
                            f"w-full items-center justify-between px-4 py-3 "
                            f"{'border-b border-slate-100' if i < len(project_summaries) - 1 else ''}"
                        ):
                            with ui.column().classes("gap-1 flex-1"):
                                with ui.row().classes("items-center gap-2"):
                                    feather("folder", size="sm", color="#64748B")
                                    ui.label(proj["project_name"]).classes(
                                        "text-sm font-medium text-slate-700 cursor-pointer hover:text-blue-600"
                                    ).on(
                                        "click", lambda pid=proj["project_id"]: ui.navigate.to(f"/project/{pid}")
                                    )
                                if proj["period_start"]:
                                    ui.label(f"Period: {proj['period_start']} to {proj['period_end']}").classes(
                                        "text-xs text-slate-400"
                                    )
                            with ui.column().classes("gap-0 items-end"):
                                ui.label(f"${proj['total_cost_usd']:.4f}").classes(
                                    "text-sm font-semibold text-slate-800"
                                )
                                ui.label(
                                    f"{proj['total_agent_calls']} calls, {proj['total_agent_tokens']:,} tokens"
                                ).classes("text-xs text-slate-500")
            else:
                with ui.card().classes(CARD_CLS + " p-6 text-center"):
                    ui.label("No project usage data yet").classes("text-slate-500")
        
        # Recent activity (right column)
        with ui.column().classes("flex-1"):
            ui.label("Recent Agent Activity").classes("text-lg font-semibold text-slate-700 mb-3")
            recent_activity = db.get_recent_agent_activity(state.user_id, limit=10)
            
            if recent_activity:
                with ui.card().classes(CARD_CLS + " p-0"):
                    for i, activity in enumerate(recent_activity):
                        status_color = "text-green-600" if activity["status"] == "success" else "text-red-600"
                        
                        with ui.row().classes(
                            f"w-full items-center justify-between px-4 py-3 "
                            f"{'border-b border-slate-100' if i < len(recent_activity) - 1 else ''}"
                        ):
                            with ui.column().classes("gap-1 flex-1"):
                                with ui.row().classes("items-center gap-2"):
                                    # Agent type icon
                                    icon_map = {
                                        "Physics": "zap",
                                        "Auditor": "shield-check", 
                                        "Generation": "edit-3",
                                        "Research": "search",
                                        "Other": "cpu"
                                    }
                                    icon = icon_map.get(activity["agent_type"], "cpu")
                                    feather(icon, size="sm", color="#64748B")
                                    
                                    ui.label(activity["agent_type"]).classes(
                                        f"text-sm font-medium {status_color}"
                                    )
                                
                                project_info = f" • {activity['project_name']}" if activity["project_name"] != "Unknown" else ""
                                ui.label(
                                    f"{activity['created_at']}{project_info}"
                                ).classes("text-xs text-slate-400")
                                
                            with ui.column().classes("gap-0 items-end"):
                                if activity["estimated_cost_usd"] > 0:
                                    ui.label(f"${activity['estimated_cost_usd']:.4f}").classes(
                                        "text-sm font-semibold text-slate-800"
                                    )
                                if activity["total_tokens"]:
                                    ui.label(f"{activity['total_tokens']:,} tokens").classes(
                                        "text-xs text-slate-500"
                                    )
                                if activity["execution_time_ms"]:
                                    ui.label(f"{activity['execution_time_ms']}ms").classes(
                                        "text-xs text-slate-400"
                                    )
            else:
                with ui.card().classes(CARD_CLS + " p-6 text-center"):
                    ui.label("No recent activity").classes("text-slate-500")