# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared SQLModel persistence layer for Shadow-Loom.

Used by both the NiceGUI web UI and the MCP server.
Standalone — no UI imports.  Configure via ``init_db(database_url)``.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlmodel import (
    Column,
    DateTime,
    Field,
    Relationship,
    Session,
    SQLModel,
    Text,
    UniqueConstraint,
    create_engine,
    or_,
    select,
)
from sqlalchemy import Index
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

# =====================================================================
# ORM Base
# =====================================================================

EXAMPLE_USER_PROVIDER_ID = "local:example"
LOCAL_USER_PROVIDER_ID = "local:default"

# Default cost-rule sentinel. Looked up by
# :meth:`shadow_loom.cost_calculation.CostCalculator.get_cost_rule` as
# the final fallback when no provider/model-specific rule exists, so
# every logged agent call lands on *some* non-zero unit price. Values
# are USD per single token (the schema's ``cost_per_unit_usd`` field
# is per-unit, not per-million); we store the per-token value so the
# arithmetic in ``calculate_agent_call_cost`` (``tokens * price``)
# returns the right dollar amount without extra scaling.
DEFAULT_COST_RULE_PROVIDER = "default"
DEFAULT_INPUT_COST_PER_TOKEN_USD = 0.039 / 1_000_000  # $0.039 / 1M input tokens
DEFAULT_OUTPUT_COST_PER_TOKEN_USD = 0.18 / 1_000_000  # $0.18 / 1M output tokens

Base = SQLModel


# =====================================================================
# Tables
# =====================================================================


class UserRow(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str = Field(max_length=32)
    provider_id: str = Field(max_length=256, unique=True)
    username: str = Field(max_length=256)
    display_name: Optional[str] = Field(default=None, max_length=256)
    email: Optional[str] = Field(default=None, max_length=256)
    avatar_url: Optional[str] = Field(default=None, max_length=512)
    bio: Optional[str] = Field(default=None, max_length=1024)
    is_example: bool = Field(default=False)
    is_admin: bool = Field(default=False)
    preferences_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    last_login_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )

    projects: list["ProjectRow"] = Relationship(back_populates="owner")


class ProjectRow(SQLModel, table=True):
    __tablename__ = "projects"
    __table_args__ = (
        # Round-13 R13-02: ``list_projects`` and every dashboard
        # query filters by ``owner_id``. Without this index the
        # access-control join scans the full projects table.
        Index("ix_projects_owner_id", "owner_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=256)
    owner_id: Optional[int] = Field(default=None, foreign_key="users.id")
    label: Optional[str] = Field(default=None, max_length=256)
    description: Optional[str] = Field(default=None, max_length=1024)
    raw_text: Optional[str] = Field(default=None, sa_column=Column(Text))
    is_public: bool = Field(default=False)
    is_template: bool = Field(default=False)
    forked_from_id: Optional[int] = Field(default=None, foreign_key="projects.id")
    star_count: int = Field(default=0)
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime,
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
    )

    owner: Optional[UserRow] = Relationship(back_populates="projects")
    versions: list["VersionRow"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"order_by": "VersionRow.version"},
    )


class VersionRow(SQLModel, table=True):
    """A single node in the version tree for a project.

    ``ancestor_id`` points to the *parent* version row (not version
    number) so that multiple branches can fork from the same ancestor.
    ``ancestor_id IS NULL`` marks the root (v0).
    """

    __tablename__ = "versions"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_project_version"),
        # Hot path: load a project's version list newest-first.
        Index("ix_versions_project_version", "project_id", "version"),
        # Walking the version DAG (ancestor lookups, branch summaries).
        Index("ix_versions_ancestor", "ancestor_id"),
        # Per-user activity / "my recent edits" feed.
        Index("ix_versions_user_created", "user_id", "created_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    version: int = Field(default=0)
    ancestor_id: Optional[int] = Field(default=None, foreign_key="versions.id")

    source: str = Field(default="ingestion", max_length=64)
    description: Optional[str] = Field(default=None, max_length=512)
    label: Optional[str] = Field(default=None, max_length=128)
    is_bookmarked: bool = Field(default=False)

    # AMWN branch metadata (Story-integration plan, Step 2). ``world_id``
    # tags every version onto either the canonical mainline
    # (``"factual"``) or a counterfactual fork (``"shadow"``).
    # ``branch_label`` is an optional human-readable name typically only
    # set on the first version of a shadow fork, e.g. "What if Duncan lived".
    world_id: str = Field(default="factual", max_length=16, index=True)
    branch_label: Optional[str] = Field(default=None, max_length=256)

    world_state_json: str = Field(sa_column=Column(Text, nullable=False))
    changeset_json: Optional[str] = Field(default=None, sa_column=Column(Text))

    # Provenance fields
    raw_query: Optional[str] = Field(default=None, sa_column=Column(Text))
    parsed_query_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    prose: Optional[str] = Field(default=None, sa_column=Column(Text))

    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )

    project: Optional["ProjectRow"] = Relationship(back_populates="versions")
    parent: Optional["VersionRow"] = Relationship(
        sa_relationship_kwargs={"remote_side": "VersionRow.id", "backref": "children"},
    )


class ProjectMemberRow(SQLModel, table=True):
    """Project-level access control for collaboration."""

    __tablename__ = "project_members"
    __table_args__ = (
        UniqueConstraint("project_id", "user_id", name="uq_project_member"),
        # Round-13 R13-02: ``list_projects`` joins through member rows
        # by ``user_id``. The unique constraint above only covers the
        # ``(project_id, user_id)`` pair leading with project_id, so a
        # by-user lookup still scans.
        Index("ix_project_members_user_id", "user_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    user_id: int = Field(foreign_key="users.id")
    role: str = Field(default="viewer", max_length=32)  # viewer | editor | admin
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class ProjectStarRow(SQLModel, table=True):
    """User bookmarks / stars for projects."""

    __tablename__ = "project_stars"
    __table_args__ = (
        UniqueConstraint("project_id", "user_id", name="uq_project_star"),
        # Round-13 R13-02: per-user "my starred projects" feeds.
        Index("ix_project_stars_user_id", "user_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    user_id: int = Field(foreign_key="users.id")
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class ApiKeyRow(SQLModel, table=True):
    """Per-user bearer tokens for MCP / API access."""

    __tablename__ = "api_keys"

    id: Optional[int] = Field(default=None, primary_key=True)
    # Round-13 R13-03: ``list_api_keys`` and ``revoke_api_key`` filter
    # by ``user_id`` on every settings-page load. The previous schema
    # only indexed ``key_hash`` (validation hot path), leaving the
    # per-user enumeration as a full table scan.
    user_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=128)
    # AUDIT (post-2026-05-26): explicit DB index on ``key_hash``.
    # ``validate_api_key`` runs on every authenticated request and
    # equality-filters on this column; without an index lookup was
    # O(N) over the entire api_keys table.
    key_hash: str = Field(max_length=128, index=True)
    key_prefix: str = Field(max_length=16)
    scopes: str = Field(default="read,write", max_length=128)
    is_active: bool = Field(default=True)
    last_used_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    expires_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class ActivityRow(SQLModel, table=True):
    """Activity feed entries for a project.

    On Postgres this table is converted to a RANGE-partitioned table on
    ``created_at`` (one partition per month) by
    :func:`ensure_pg_partitions` — either automatically when empty
    (fresh deploys) or via ``scripts/setup_pg_partitions.py --migrate``
    when it already contains data. The SQLModel definition itself is
    kept dialect-portable (single ``id`` PK, nullable ``created_at``)
    so SQLite remains a first-class backend.
    """

    __tablename__ = "activities"
    __table_args__ = (
        # Project feed: most queries are "latest N activities for project X".
        Index("ix_activities_project_created", "project_id", "created_at"),
        # Per-user activity feeds (profile pages, audit logs).
        Index("ix_activities_user_created", "user_id", "created_at"),
        # Cheap join lookup when surfacing the activity row that owns a
        # given version (UI: "this version was created by …").
        Index("ix_activities_version", "version_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    action: str = Field(max_length=64)  # ingestion | query | edit | share | fork | star
    summary: Optional[str] = Field(default=None, max_length=512)
    detail_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    version_id: Optional[int] = Field(default=None, foreign_key="versions.id")
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class ActiveVersionRow(SQLModel, table=True):
    """Persistent per-user-per-project active version pointer.

    Used by the MCP service so a sequence of tool calls can default to
    the version the user has selected (e.g. via the UI version-tree
    sidebar) rather than always falling back to ``latest``.
    """

    __tablename__ = "active_versions"
    __table_args__ = (
        UniqueConstraint("project_id", "user_id", name="uq_active_version_user_project"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    user_id: int = Field(foreign_key="users.id")
    version_row_id: int = Field(foreign_key="versions.id")
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime,
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
    )


class ResearchCacheRow(SQLModel, table=True):
    """Cached output of a ``ResearchProvider.search`` call.

    Keyed by ``key = sha256(provider | provider_model | query)`` so
    identical re-runs are deterministic and free. Snippets are stored
    as a JSON-serialised list of ``shadow_loom.research.ResearchSnippet``
    payloads in a Text column (matching the project's existing
    ``changeset_json`` / ``world_state_json`` convention).

    Per-account isolation: ``user_id`` is part of the uniqueness key so
    one user's cached lookups are never served to another. Set
    ``user_id`` to ``None`` only for shared, system-level fixtures
    (which the production code paths never do).
    """

    __tablename__ = "research_cache"
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_research_cache_user_key"),
        # Lookups by (user, provider, model) when warming the cache page
        # or invalidating after a provider config change.
        Index(
            "ix_research_cache_user_provider",
            "user_id", "provider", "provider_model",
        ),
        # Recency-ordered scans (cleanup, dashboards).
        Index("ix_research_cache_created_at", "created_at"),
        # NOTE: ``research_cache`` is intentionally *not* partitioned.
        # Postgres requires every unique constraint on a partitioned
        # table to include the partition key, which would weaken the
        # ``(user_id, key)`` dedup invariant the cache depends on.
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    key: str = Field(max_length=64, index=True)
    provider: str = Field(max_length=32)
    provider_model: str = Field(default="", max_length=64)
    query: str = Field(sa_column=Column(Text, nullable=False))
    snippets_json: str = Field(sa_column=Column(Text, nullable=False))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class WorldFactRow(SQLModel, table=True):
    """Persistent per-project ``WorldFact`` record.

    Mirrors ``shadow_loom.research.WorldFact``. Stored separately from
    ``VersionRow.world_state_json`` so facts persist across versions
    without bloating every snapshot, and so the UI can list / edit /
    delete them independently of the version tree. Each version's
    serialised ``WorldStateV1.world_facts`` is rebuilt from this table
    on save (Phase 2 wiring).
    """

    __tablename__ = "world_facts"
    __table_args__ = (
        UniqueConstraint("project_id", "fact_id", name="uq_world_fact_project_id"),
        # Recency-ordered listing within a project (UI inspector).
        Index("ix_world_facts_project_retrieved", "project_id", "retrieved_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    fact_id: str = Field(max_length=128)  # the FACT_ id
    topic: str = Field(sa_column=Column(Text, nullable=False))
    summary: str = Field(sa_column=Column(Text, nullable=False))
    confidence: str = Field(default="moderate", max_length=16)
    source_url_primary: Optional[str] = Field(default=None, sa_column=Column(Text))
    provider: str = Field(default="", max_length=32)
    related_node_ids_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    raw_snippets_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    retrieved_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class ProjectSettingsRow(SQLModel, table=True):
    """Per-project settings that live outside ``WorldStateV1``.

    Currently holds the project-scoped ``research_topics`` list used by
    the research-extraction pipeline (Step 3d). Kept off the world model
    deliberately: research config is not narrative state, must not fork
    with shadow branches, and must not bloat every version snapshot.
    """

    __tablename__ = "project_settings"

    project_id: int = Field(
        primary_key=True,
        foreign_key="projects.id",
    )
    research_topics_json: str = Field(default="[]", sa_column=Column(Text))
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime,
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
    )


class UserModelSettingsRow(SQLModel, table=True):
    """Per-user LLM provider & model preferences.

    Lets each signed-in user override the deployment's ``DEFAULT_MODEL``
    env var (and any per-stage ``*_MODEL`` env var) from the Settings
    page, and register their own OpenAI-compatible providers — e.g. a
    private llama.cpp endpoint, a paid Mistral key, or a self-hosted
    vLLM cluster. The values are merged into the resolver via
    :func:`shadow_loom.settings.set_user_context`.

    Stored as JSON blobs to avoid schema churn when adding new stages
    or provider fields.
    """

    __tablename__ = "user_model_settings"

    user_id: int = Field(
        primary_key=True,
        foreign_key="users.id",
    )
    # Empty string → fall back to env var.
    default_model: str = Field(default="")
    # JSON object: {"generation": "openrouter:...", "auditor": "...", ...}.
    # Recognised stage keys: generation, auditor, auditor_generation,
    # extraction, query_parsing.
    stage_models_json: str = Field(default="{}", sa_column=Column(Text))
    # JSON array of {"prefix": str, "base_url": str, "api_key": str,
    # "is_local": bool}. Prefix is lowercased and used as the
    # ``<prefix>:`` model-string scheme.
    custom_providers_json: str = Field(default="[]", sa_column=Column(Text))
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime,
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
    )


class AgentCallLogRow(SQLModel, table=True):
    """Log every agent execution with performance and cost tracking.

    Postgres-partitioned by month on ``created_at`` — see
    :class:`ActivityRow` for the partitioning lifecycle.
    """
    __tablename__ = "agent_call_logs"
    __table_args__ = (
        # Per-user dashboards / billing rollups: "show calls for user X
        # in the last 30 days, newest first".
        Index("ix_agent_call_logs_user_created", "user_id", "created_at"),
        # Per-project cost & timing breakdowns.
        Index("ix_agent_call_logs_project_created", "project_id", "created_at"),
        # Group-by agent_type queries used by the cost analytics panel.
        Index("ix_agent_call_logs_agent_type", "agent_type"),
        # Cross-reference to Langfuse traces.
        Index("ix_agent_call_logs_langfuse_trace", "langfuse_trace_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    project_id: Optional[int] = Field(default=None, foreign_key="projects.id") 
    version_id: Optional[int] = Field(default=None, foreign_key="versions.id")
    
    # Agent identification
    agent_type: str = Field(max_length=64)  # "Physics", "Auditor", "Generation", etc.
    agent_name: str = Field(max_length=128)  # Full agent name/class
    
    # Execution details  
    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)
    total_tokens: Optional[int] = Field(default=None)
    
    # Performance metrics
    execution_time_ms: int = Field(default=0)
    model_provider: Optional[str] = Field(default=None, max_length=32)
    model_name: Optional[str] = Field(default=None, max_length=64)
    
    # Status tracking
    status: str = Field(default="success", max_length=16)  # success, error, timeout
    error_message: Optional[str] = Field(default=None, max_length=512)
    
    # Cost calculation (populated by background job)
    estimated_cost_usd: Optional[float] = Field(default=None)
    
    # Metadata & provenance
    langfuse_trace_id: Optional[str] = Field(default=None, max_length=128)
    metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )
    
    # Relationships
    user: UserRow = Relationship()
    project: Optional[ProjectRow] = Relationship()
    version: Optional[VersionRow] = Relationship()


class ApiCallLogRow(SQLModel, table=True):
    """Track external API calls (Tavily, etc.) for cost analysis.

    Postgres-partitioned by month on ``created_at`` — see
    :class:`ActivityRow` for the partitioning lifecycle.
    """
    __tablename__ = "api_call_logs"
    __table_args__ = (
        Index("ix_api_call_logs_user_created", "user_id", "created_at"),
        Index("ix_api_call_logs_project_created", "project_id", "created_at"),
        # Cost rollups by provider/service.
        Index(
            "ix_api_call_logs_provider_created",
            "provider", "service_type", "created_at",
        ),
        # Join back to the agent call that triggered this API call.
        Index("ix_api_call_logs_agent_call", "agent_call_log_id"),
    )
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id") 
    project_id: Optional[int] = Field(default=None, foreign_key="projects.id")
    version_id: Optional[int] = Field(default=None, foreign_key="versions.id")
    
    # API identification
    provider: str = Field(max_length=32)  # "tavily", "openai", "anthropic"
    service_type: str = Field(max_length=32)  # "search", "llm_chat", "embeddings"
    endpoint: Optional[str] = Field(default=None, max_length=256)
    
    # Request details
    request_size: Optional[int] = Field(default=None)  # tokens, queries, etc.
    response_size: Optional[int] = Field(default=None)
    
    # Provider-specific metrics
    search_depth: Optional[str] = Field(default=None, max_length=16)  # For Tavily
    results_count: Optional[int] = Field(default=None)  # For search APIs
    
    # Performance & status
    response_time_ms: int = Field(default=0)
    status_code: Optional[int] = Field(default=None)
    status: str = Field(default="success", max_length=16)
    error_message: Optional[str] = Field(default=None, max_length=512)
    
    # Cost calculation
    estimated_cost_usd: Optional[float] = Field(default=None)
    
    # Context & metadata. ``agent_call_log_id`` is intentionally *not* a
    # foreign key: Postgres forbids FKs into partitioned tables unless
    # the partition key is included on both sides, and the analytics
    # join is cheap enough with the index alone.
    agent_call_log_id: Optional[int] = Field(default=None) 
    metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )
    
    # Relationships  
    user: UserRow = Relationship()
    project: Optional[ProjectRow] = Relationship() 
    version: Optional[VersionRow] = Relationship()


class CostRuleRow(SQLModel, table=True):
    """Define pricing rules for different services and models."""
    __tablename__ = "cost_rules"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Service identification  
    provider: str = Field(max_length=32)  # "openai", "tavily", "anthropic"
    service_type: str = Field(max_length=32)  # "llm_chat", "search", "embeddings"
    model_name: Optional[str] = Field(default=None, max_length=64)  # "gpt-4o", "claude-3"
    
    # Pricing structure
    unit_type: str = Field(max_length=16)  # "tokens", "requests", "results"
    cost_per_unit_usd: float = Field(default=0.0)
    
    # Input/output pricing (for LLMs)  
    input_cost_per_unit_usd: Optional[float] = Field(default=None)
    output_cost_per_unit_usd: Optional[float] = Field(default=None)
    
    # Tier-based pricing  
    tier_threshold: Optional[int] = Field(default=None)
    tier_cost_per_unit_usd: Optional[float] = Field(default=None)
    
    # Rule metadata
    description: Optional[str] = Field(default=None, max_length=256)
    effective_from: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )
    effective_to: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class UserUsageSummaryRow(SQLModel, table=True):
    """Rollup usage statistics per user for dashboard/billing."""
    __tablename__ = "user_usage_summaries"
    __table_args__ = (
        UniqueConstraint("user_id", "period_start", name="uq_user_period"),
    )
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    
    # Time period (daily/monthly rollups)
    period_type: str = Field(max_length=16)  # "daily", "monthly", "lifetime"
    period_start: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    period_end: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    
    # Agent usage rollups
    total_agent_calls: int = Field(default=0)
    total_agent_tokens: int = Field(default=0) 
    total_agent_cost_usd: float = Field(default=0.0)
    
    # API usage rollups
    total_api_calls: int = Field(default=0)
    total_api_cost_usd: float = Field(default=0.0)
    
    # Top agent types (JSON)
    agent_type_breakdown_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Top providers (JSON) 
    provider_breakdown_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )
    
    user: UserRow = Relationship()


class ProjectUsageSummaryRow(SQLModel, table=True):
    """Rollup usage statistics per project for analysis."""
    __tablename__ = "project_usage_summaries" 
    __table_args__ = (
        UniqueConstraint("project_id", "period_start", name="uq_project_period"),
    )
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    
    # Time period
    period_type: str = Field(max_length=16)  # "daily", "monthly", "lifetime"
    period_start: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    period_end: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    
    # Usage metrics
    total_agent_calls: int = Field(default=0)
    total_agent_tokens: int = Field(default=0)
    total_agent_cost_usd: float = Field(default=0.0)
    total_api_calls: int = Field(default=0)
    total_api_cost_usd: float = Field(default=0.0)
    
    # Version count for context
    versions_created: int = Field(default=0)
    
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )
    
    project: ProjectRow = Relationship()


class McpIdempotencyRow(SQLModel, table=True):
    """Round-9 C7: client-supplied idempotency cache for MCP write tools.

    ``run_and_save`` consults this table before invoking the pipeline
    when an ``idempotency_key`` is supplied. A match replays the
    previously cached response envelope without re-running the
    pipeline, so client-side retries (network blip, MCP transport
    reconnect, etc.) don't produce duplicate version rows or burn
    additional generation budget.

    ``key_hash`` is ``sha256(project_id|ancestor_id|idempotency_key)``
    so the same client-supplied key on different projects / ancestors
    yields distinct rows.
    """
    __tablename__ = "mcp_idempotency"
    __table_args__ = (
        Index("ix_mcp_idempotency_created", "created_at"),
    )

    key_hash: str = Field(primary_key=True, max_length=64)
    project_id: int = Field(foreign_key="projects.id", index=True)
    ancestor_id: Optional[int] = Field(default=None, index=True)
    version_row_id: Optional[int] = Field(default=None, foreign_key="versions.id")
    response_json: str = Field(sa_column=Column(Text, nullable=False))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, default=lambda: datetime.now(timezone.utc)),
    )


class SchemaVersionRow(SQLModel, table=True):
    """Lightweight schema-version ledger.

    Records every additive migration applied by
    :func:`_run_lightweight_migrations` so operators can audit which
    migrations have run on a given database without reverse-engineering
    column lists. This is *not* Alembic — it does not generate
    migrations or support down-revisions — but it gives us a paper
    trail and a place to read the current schema version from
    administrative tooling.

    The current schema version is the maximum ``version`` value present.
    A fresh database (post ``create_all``) is bootstrapped to the
    pinned :data:`SCHEMA_VERSION_CURRENT` so future migrations only
    apply deltas.
    """
    __tablename__ = "schema_versions"

    version: int = Field(primary_key=True)
    name: str = Field(max_length=128)
    applied_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# Bump when adding a new entry to ``_SCHEMA_MIGRATIONS`` below.
SCHEMA_VERSION_CURRENT: int = 4

# Ordered ledger of applied migrations: (version, name).
# Version 1 is the historical baseline (everything before this ledger
# existed); version 2 added world_id + branch_label columns to
# ``versions`` (handled by ``_run_lightweight_migrations``); version 3
# introduced Postgres RANGE partitioning + composite indexes on the
# high-volume log tables (activities, agent_call_logs, api_call_logs);
# version 4 (round-13 R13-02/03) added per-user indexes on the hot
# access-control tables (projects.owner_id, project_members.user_id,
# project_stars.user_id, api_keys.user_id).
_SCHEMA_MIGRATIONS: list[tuple[int, str]] = [
    (1, "baseline"),
    (2, "versions.world_id+branch_label"),
    (3, "log-tables.partitioning+indexes"),
    (4, "user-access.indexes"),
]


# =====================================================================
# Engine / Session factory
# =====================================================================

_engine = None
# D6 (thirteenth-pass audit): ``init_db`` used to perform an unguarded
# ``_engine = create_engine(...)`` followed by ``create_all`` /
# migrations / partition setup. Under the multi-tenant NiceGUI app
# multiple worker threads can race on first request and each call
# ``init_db`` concurrently \u2014 the second thread would clobber the
# first engine mid-migration, leaking connections and intermittently
# corrupting the schema-version stamp. Guard the whole bootstrap with
# a double-checked lock so only the first caller runs migrations and
# subsequent callers no-op once ``_engine`` is set.
import threading as _threading

_engine_lock = _threading.Lock()


def get_engine():
    global _engine
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _engine


def get_session() -> Session:
    return Session(get_engine())


def init_db(database_url: str = "sqlite:///shadow_loom.db") -> None:
    """Create the engine, all tables, and seed the example user."""
    global _engine

    # D6 (thirteenth-pass audit): serialise concurrent first-call
    # initialisation. We do NOT short-circuit when ``_engine`` is
    # already set \u2014 several test fixtures rely on
    # ``init_db("sqlite://")`` *replacing* the engine on every test to
    # get a clean in-memory shard, and silently no-op'ing would
    # smuggle stale state across tests. The lock only prevents two
    # threads from racing inside ``_init_db_locked`` and corrupting
    # the migration / partition-setup steps.
    with _engine_lock:
        _init_db_locked(database_url)


def _init_db_locked(database_url: str) -> None:
    """Body of :func:`init_db`, executed under ``_engine_lock``."""
    global _engine

    # Railway / Heroku-style ``postgres://`` URLs are legacy SQLAlchemy
    # syntax — rewrite to the modern ``postgresql+psycopg://`` driver
    # string so the same env var works on every host.
    if database_url.startswith("postgres://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
    elif database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]

    engine_kwargs: dict = {"echo": False}
    if database_url.startswith("postgresql"):
        # Tuned for a small Railway dyno + a handful of concurrent
        # NiceGUI sessions. ``pool_pre_ping`` survives idle drops.
        engine_kwargs.update(
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=1800,
        )
    elif database_url.startswith("sqlite"):
        # The UI/MCP layers run DB calls inside ``asyncio.to_thread``
        # workers, so a file-backed connection from the default
        # ``QueuePool`` can be checked out on a different thread than the
        # one that opened it. pysqlite forbids that by default
        # (``check_same_thread=True``) → intermittent
        # ``sqlite3.ProgrammingError`` under concurrency. Allow
        # cross-thread use. We deliberately leave ``poolclass`` at its
        # SQLAlchemy default (QueuePool for files, SingletonThreadPool
        # for in-memory): pinning a single shared StaticPool connection
        # would serialise onto one cursor and corrupt under concurrent
        # writes.
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    _engine = create_engine(database_url, **engine_kwargs)
    # Enable SQLite foreign-key enforcement so the schema's FK
    # declarations and our delete-ordering invariants are validated
    # at runtime (SQLite leaves FKs OFF by default).
    if database_url.startswith("sqlite"):
        from sqlalchemy import event as _sa_event

        @_sa_event.listens_for(_engine, "connect")
        def _enable_sqlite_fk(dbapi_conn, _conn_record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA foreign_keys=ON")
            finally:
                cur.close()

    SQLModel.metadata.create_all(_engine)
    _run_lightweight_migrations(_engine)
    partition_max_version: Optional[int] = None
    if database_url.startswith("postgresql"):
        # Round-10 R10-07: previously partition setup failures were
        # swallowed inside ``ensure_pg_partitions`` (logger.exception
        # + continue), and ``_record_schema_versions`` then stamped
        # version 3 unconditionally — giving operators a false-green
        # migration record while the cluster was still running on the
        # un-partitioned tables. The helper now returns a success
        # bool. On failure we either raise (fail-fast default) or
        # explicitly skip the v3 stamp when the operator opts in to
        # degraded mode via ``SHADOW_LOOM_ALLOW_PARTITION_FAIL=1``.
        ok = ensure_pg_partitions(_engine)
        if not ok:
            import os as _os
            if _os.environ.get("SHADOW_LOOM_ALLOW_PARTITION_FAIL", "").strip().lower() not in {"1", "true", "yes"}:
                raise RuntimeError(
                    "[DB] Postgres partition setup failed; refusing to "
                    "stamp schema version 3. Inspect the [DB·partitions] "
                    "log entries above and re-run "
                    "scripts/setup_pg_partitions.py --migrate. To start "
                    "anyway in degraded mode, set "
                    "SHADOW_LOOM_ALLOW_PARTITION_FAIL=1 (the partitioning "
                    "migration will NOT be stamped, so the next startup "
                    "will retry)."
                )
            logger.warning(
                "[DB] SHADOW_LOOM_ALLOW_PARTITION_FAIL is set; "
                "continuing without stamping schema version 3."
            )
            partition_max_version = 2  # cap stamping below v3
    _record_schema_versions(_engine, max_version=partition_max_version)
    ensure_example_user()
    ensure_default_cost_rule()
    try:
        removed = purge_mcp_idempotency()
        if removed:
            logger.info(
                "[DB] Purged %s stale mcp_idempotency rows on startup.", removed
            )
    except Exception:
        logger.debug("[DB] mcp_idempotency purge skipped", exc_info=True)
    # Redact credentials from the DSN before logging \u2014 Postgres
    # connection strings carry ``user:password@host`` which would
    # leak to shared log aggregators (round-3 audit).
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(database_url)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            netloc = f"***@{host}" if host else "***"
            redacted = urlunparse((
                parsed.scheme, netloc, parsed.path,
                parsed.params, "", parsed.fragment,
            ))
        else:
            redacted = database_url
    except Exception:  # noqa: BLE001
        # Fall back to scheme-only so we never leak on a parse failure.
        redacted = database_url.split("://", 1)[0] + "://***"
    logger.info("[DB] Tables initialised on %s", redacted)


def _record_schema_versions(engine, *, max_version: Optional[int] = None) -> None:  # noqa: ANN001
    """Stamp ``schema_versions`` so we can audit migration history.

    Inserts every entry from :data:`_SCHEMA_MIGRATIONS` that isn't
    already present. Idempotent: re-running just no-ops, so it's safe
    to call on every startup.

    Round-10 R10-07: when *max_version* is set, migrations strictly
    above that version are skipped. Used by ``init_db`` to avoid
    stamping ``log-tables.partitioning+indexes`` (version 3) when
    Postgres partition setup failed and the operator chose degraded
    mode — the next clean startup will then complete the stamping.
    """
    with Session(engine) as s:
        existing = set(s.exec(select(SchemaVersionRow.version)).all())
        added = 0
        for version, name in _SCHEMA_MIGRATIONS:
            if version in existing:
                continue
            if max_version is not None and version > max_version:
                continue
            s.add(SchemaVersionRow(version=version, name=name))
            added += 1
        if added:
            s.commit()
            logger.info(
                "[DB·migrate] Stamped %d schema version row(s); current=%d.",
                added, SCHEMA_VERSION_CURRENT,
            )


def _run_lightweight_migrations(engine) -> None:  # noqa: ANN001
    """Apply additive column migrations that ``create_all`` cannot handle.

    ``SQLModel.metadata.create_all`` only creates *missing* tables; it
    never adds new columns to existing ones. This helper inspects the
    live ``versions`` table and adds any of the AMWN branch metadata
    columns that pre-date their introduction (Story-integration plan,
    Step 2). Idempotent: each ALTER is guarded by a column-existence
    check so the helper is safe to run on every startup.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "versions" not in insp.get_table_names():
        return
    existing_cols = {c["name"] for c in insp.get_columns("versions")}
    backend = engine.dialect.name  # "sqlite" | "postgresql" | ...

    statements: list[str] = []
    if "world_id" not in existing_cols:
        # Both SQLite and Postgres accept this exact form; existing rows
        # are backfilled to 'factual' which matches WorldModelVersion's
        # default and preserves the pre-AMWN linear semantics.
        statements.append(
            "ALTER TABLE versions ADD COLUMN world_id VARCHAR(16) "
            "NOT NULL DEFAULT 'factual'"
        )
        # Index the new column so the version-tree queries stay cheap.
        if backend == "sqlite":
            statements.append(
                "CREATE INDEX IF NOT EXISTS ix_versions_world_id "
                "ON versions (world_id)"
            )
        elif backend == "postgresql":
            statements.append(
                "CREATE INDEX IF NOT EXISTS ix_versions_world_id "
                "ON versions (world_id)"
            )
    if "branch_label" not in existing_cols:
        statements.append(
            "ALTER TABLE versions ADD COLUMN branch_label VARCHAR(256)"
        )

    # Round-13 R13-02/03: ensure the per-user access-control indexes
    # exist on legacy databases that pre-date schema version 4.
    # ``CREATE INDEX IF NOT EXISTS`` is identical syntax on SQLite
    # and Postgres so a single statement list works on both backends.
    # Each index name matches the declarative ``Index(...)`` in the
    # SQLModel definition so ``create_all`` on a fresh DB also lands
    # the same physical name.
    _r13_index_specs: list[tuple[str, str, str]] = [
        ("ix_projects_owner_id", "projects", "owner_id"),
        ("ix_project_members_user_id", "project_members", "user_id"),
        ("ix_project_stars_user_id", "project_stars", "user_id"),
        ("ix_api_keys_user_id", "api_keys", "user_id"),
    ]
    _existing_tables = set(insp.get_table_names())
    for idx_name, tbl, col in _r13_index_specs:
        if tbl not in _existing_tables:
            continue
        statements.append(
            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl} ({col})"
        )

    if not statements:
        return

    # Round-6 audit: only swallow the specific "already exists" race
    # raised by concurrent migrators; let real DDL errors propagate so
    # the boot fails loudly rather than silently leaving the schema
    # half-migrated. Re-inspect after each ALTER so a racing peer that
    # already added the column is treated as a no-op.
    #
    # Round-11 R11-10: wrap the whole batch in a single outer
    # transaction with per-statement SAVEPOINTs so the schema either
    # advances atomically or rolls back to the pre-migration state.
    # The previous per-statement ``engine.begin()`` loop could leave
    # the schema half-migrated if a later statement raised a non-
    # benign error after earlier ones had already committed — making
    # subsequent boots fail with confusing "column X exists but
    # column Y does not" errors. With SAVEPOINTs a benign duplicate
    # rolls back just that statement while preserving the prior
    # successful adds in the outer transaction.
    benign_fragments = (
        "duplicate column",
        "already exists",
    )
    with engine.begin() as conn:
        for sql in statements:
            sp = conn.begin_nested()
            try:
                conn.execute(text(sql))
                sp.commit()
            except Exception as exc:  # noqa: BLE001
                sp.rollback()
                msg = str(exc).lower()
                if any(frag in msg for frag in benign_fragments):
                    logger.info(
                        "[DB\u00b7migrate] Skipped (already applied by peer): %s",
                        sql,
                    )
                    continue
                # Re-inspect to handle dialects whose error message
                # doesn't include either fragment.
                fresh = {c["name"] for c in inspect(engine).get_columns("versions")}
                if "ADD COLUMN" in sql.upper():
                    parts = sql.upper().split("ADD COLUMN", 1)[1].strip().split()
                    if parts and parts[0].lower() in fresh:
                        logger.info(
                            "[DB\u00b7migrate] Skipped (column already present): %s",
                            sql,
                        )
                        continue
                logger.error(
                    "[DB\u00b7migrate] Failed to apply: %s \u2014 rolling back batch.",
                    sql,
                )
                raise
    logger.info(
        "[DB·migrate] Applied %d additive column migration(s) to 'versions'.",
        len(statements),
    )


# =====================================================================
# Postgres declarative partitioning
# =====================================================================

# High-volume log tables that are RANGE-partitioned by month on
# ``created_at`` on Postgres. The SQLModel definitions for these tables
# stay dialect-portable (single ``id`` PK, nullable ``created_at``) so
# SQLite remains usable; the partitioning is applied at the DDL level
# by :func:`ensure_pg_partitions` after ``create_all`` has run.
#
# Postgres requires the partition key to be part of every UNIQUE / PK
# constraint, so the partitioned tables created here use a composite
# ``(id, created_at)`` primary key with ``id`` as a SERIAL/IDENTITY.
_PARTITIONED_TABLES: tuple[str, ...] = (
    "activities",
    "agent_call_logs",
    "api_call_logs",
)


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    """Return ``(start, end)`` ISO timestamps for a given month."""
    if month == 12:
        nxt_year, nxt_month = year + 1, 1
    else:
        nxt_year, nxt_month = year, month + 1
    return (
        f"{year:04d}-{month:02d}-01 00:00:00",
        f"{nxt_year:04d}-{nxt_month:02d}-01 00:00:00",
    )


def _months_window(months_back: int, months_forward: int) -> list[tuple[int, int]]:
    """Return inclusive (year, month) list around the current month."""
    now = datetime.now(timezone.utc)
    months: list[tuple[int, int]] = []
    yy, mm = now.year, now.month
    for _ in range(months_back):
        mm -= 1
        if mm == 0:
            mm = 12
            yy -= 1
        months.append((yy, mm))
    months.reverse()
    months.append((now.year, now.month))
    yy, mm = now.year, now.month
    for _ in range(months_forward):
        mm += 1
        if mm == 13:
            mm = 1
            yy += 1
        months.append((yy, mm))
    return months


def _table_relkind(conn, table: str) -> Optional[str]:  # noqa: ANN001
    """Return ``relkind`` for ``table`` in the current schema, or None."""
    from sqlalchemy import text

    row = conn.execute(
        text(
            "SELECT relkind FROM pg_class WHERE relname = :name AND "
            "relnamespace = (SELECT oid FROM pg_namespace "
            "WHERE nspname = current_schema())"
        ),
        {"name": table},
    ).first()
    return row[0] if row else None


def _convert_table_to_partitioned(
    conn,  # noqa: ANN001
    table: str,
    *,
    partition_key: str = "created_at",
) -> None:
    """Drop a regular Postgres table and recreate it as partitioned.

    Caller must have verified the table is empty. The replacement
    table inherits the column shape of the original via SQLModel
    metadata, but with:

    - composite PK ``(id, <partition_key>)`` (Postgres requirement);
    - ``id`` as a ``GENERATED BY DEFAULT AS IDENTITY`` column;
    - ``<partition_key>`` declared ``NOT NULL`` with a server-side
      ``now()`` default so application code that omits the column
      keeps working.

    Foreign keys and indexes are then re-created from the SQLModel
    table object so partitions inherit them automatically.
    """
    from sqlalchemy import text

    sqla_table = SQLModel.metadata.tables.get(table)
    if sqla_table is None:
        raise RuntimeError(f"{table!r} not registered in SQLModel.metadata")

    # Build column DDL fragments preserving order.
    col_fragments: list[str] = []
    for col in sqla_table.columns:
        name = col.name
        if name == "id":
            frag = f'"id" BIGINT GENERATED BY DEFAULT AS IDENTITY'
        elif name == partition_key:
            frag = f'"{name}" TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now()'
        else:
            type_sql = col.type.compile(dialect=conn.dialect)
            null_sql = "" if col.nullable else " NOT NULL"
            frag = f'"{name}" {type_sql}{null_sql}'
        col_fragments.append(frag)

    # Composite PK including the partition key.
    col_fragments.append(f'PRIMARY KEY ("id", "{partition_key}")')

    # Inline FK definitions (Postgres allows them on partitioned tables
    # *from* the partitioned table; FKs *into* a partitioned table are
    # what's forbidden, which is why we already dropped the
    # api_call_logs.agent_call_log_id FK above).
    for fk in sqla_table.foreign_key_constraints:
        local_cols = ", ".join(f'"{c.name}"' for c in fk.columns)
        ref_table = fk.referred_table.name
        ref_cols = ", ".join(f'"{e.column.name}"' for e in fk.elements)
        col_fragments.append(
            f'FOREIGN KEY ({local_cols}) REFERENCES "{ref_table}" ({ref_cols})'
        )

    body = ",\n    ".join(col_fragments)
    create_sql = (
        f'CREATE TABLE "{table}" (\n    {body}\n) '
        f'PARTITION BY RANGE ("{partition_key}")'
    )

    conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
    conn.execute(text(create_sql))

    # Re-create indexes declared on the SQLModel table (Index objects
    # propagate to every partition automatically once attached to the
    # parent partitioned table).
    for idx in sqla_table.indexes:
        idx.create(bind=conn)


def ensure_pg_partitions(
    engine,  # noqa: ANN001
    *,
    months_back: int = 3,
    months_forward: int = 3,
) -> bool:
    """Ensure log tables are partitioned and have monthly child tables.

    Returns ``True`` when every step for every table in
    :data:`_PARTITIONED_TABLES` succeeded, ``False`` when any step
    raised. Round-10 R10-07: ``init_db`` reads this return value and
    refuses to stamp schema version 3 on failure, so a botched
    migration is visible at startup instead of leaving the cluster
    running on un-partitioned log tables with a false-green ledger.

    For each table in :data:`_PARTITIONED_TABLES` this:

    1. If the table doesn't exist as partitioned yet AND is empty,
       silently converts it in place via
       :func:`_convert_table_to_partitioned`. This is the fresh-deploy
       path — ``SQLModel.metadata.create_all`` ran first and built a
       regular table; we replace it with a partitioned one before any
       data is written.
    2. If the table exists but is non-partitioned AND non-empty,
       logs a warning and skips. Operators must run
       ``scripts/setup_pg_partitions.py --migrate`` (which copies data
       through a ``_legacy`` table) during a maintenance window.
    3. Once the table is partitioned, creates a ``DEFAULT`` partition
       (catches rows outside declared ranges) and monthly RANGE
       partitions for the window ``[now - months_back, now + months_forward]``.

    Idempotent and safe to call on every startup.
    """
    from sqlalchemy import text

    months = _months_window(months_back, months_forward)
    all_ok = True  # Round-10 R10-07: track success across all tables.

    with engine.begin() as conn:
        for table in _PARTITIONED_TABLES:
            relkind = _table_relkind(conn, table)
            if relkind is None:
                # Table not created yet — nothing to do.
                continue

            if relkind != "p":
                # Try to auto-convert if empty (fresh-deploy path).
                # Round-10 R10-08: take an ACCESS EXCLUSIVE lock first
                # so no concurrent writer can sneak rows in between the
                # count() check and the DROP TABLE that
                # ``_convert_table_to_partitioned`` performs. Without
                # the lock the count→drop window is a data-loss race
                # under READ COMMITTED isolation.
                try:
                    conn.execute(text(f'LOCK TABLE "{table}" IN ACCESS EXCLUSIVE MODE'))
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[DB·partitions] Could not LOCK %r; refusing to "
                        "auto-convert.",
                        table,
                    )
                    all_ok = False
                    continue
                count_row = conn.execute(
                    text(f'SELECT count(*) FROM "{table}"')
                ).scalar_one()
                if count_row == 0:
                    try:
                        _convert_table_to_partitioned(conn, table)
                        logger.info(
                            "[DB·partitions] Converted empty table %r to "
                            "RANGE-partitioned (by created_at).",
                            table,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[DB·partitions] Failed to auto-convert %r; "
                            "leaving as-is.",
                            table,
                        )
                        all_ok = False
                        continue
                else:
                    logger.warning(
                        "[DB·partitions] Table %r has %d row(s) and is "
                        "NOT partitioned. Run "
                        "scripts/setup_pg_partitions.py --migrate to "
                        "convert (requires a maintenance window).",
                        table, count_row,
                    )
                    all_ok = False
                    continue

            # 1. Default partition catches anything outside the declared ranges.
            default_name = f"{table}_default"
            try:
                conn.execute(
                    text(
                        f'CREATE TABLE IF NOT EXISTS "{default_name}" '
                        f'PARTITION OF "{table}" DEFAULT'
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[DB·partitions] Failed to create default partition for %s",
                    table,
                )
                all_ok = False

            # 2. Monthly partitions.
            for yy, mm in months:
                part_name = f"{table}_y{yy:04d}m{mm:02d}"
                start, end = _month_bounds(yy, mm)
                try:
                    conn.execute(
                        text(
                            f'CREATE TABLE IF NOT EXISTS "{part_name}" '
                            f'PARTITION OF "{table}" '
                            f"FOR VALUES FROM ('{start}') TO ('{end}')"
                        )
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[DB·partitions] Failed to create partition %s "
                        "for range [%s, %s)",
                        part_name, start, end,
                    )
                    all_ok = False

    logger.info(
        "[DB·partitions] Ensured monthly partitions for %d table(s) "
        "(window: -%d / +%d months; ok=%s).",
        len(_PARTITIONED_TABLES), months_back, months_forward, all_ok,
    )
    return all_ok


# =====================================================================
# Example user
# =====================================================================


def ensure_example_user() -> UserRow:
    """Create (or return) the built-in *example* user.

    Projects owned by this user are visible to everyone via
    ``list_projects()``.
    """
    with get_session() as s:
        row = s.exec(select(UserRow).where(UserRow.provider_id == EXAMPLE_USER_PROVIDER_ID)).first()
        if row is None:
            row = UserRow(
                provider="local",
                provider_id=EXAMPLE_USER_PROVIDER_ID,
                username="example",
                is_example=True,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
        return row


def get_example_user_id() -> Optional[int]:
    """Return the example user's row id, or None if not yet created."""
    with get_session() as s:
        row = s.exec(select(UserRow).where(UserRow.provider_id == EXAMPLE_USER_PROVIDER_ID)).first()
        return row.id if row else None


def ensure_default_cost_rule() -> "CostRuleRow":
    """Create (or return) the catch-all default cost rule.

    Without a row in ``cost_rules`` matching the call's provider /
    model, :meth:`CostCalculator.get_cost_rule` returns ``None`` and
    every logged agent call is stamped ``estimated_cost_usd = 0.0``
    — the per-user / per-project rollups in ``UserUsageSummaryRow``
    and ``ProjectUsageSummaryRow`` then sum to zero too.

    This helper seeds a single ``provider="default"`` rule
    (input $0.039 / 1M tokens, output $0.18 / 1M tokens) used by the
    calculator's *final* fallback lookup once no provider-specific
    rule matches. Idempotent: re-running returns the existing row.
    """
    with get_session() as s:
        row = s.exec(
            select(CostRuleRow).where(
                CostRuleRow.provider == DEFAULT_COST_RULE_PROVIDER,
                CostRuleRow.service_type == "llm_chat",
                CostRuleRow.model_name.is_(None),
            )
        ).first()
        if row is None:
            row = CostRuleRow(
                provider=DEFAULT_COST_RULE_PROVIDER,
                service_type="llm_chat",
                model_name=None,
                unit_type="tokens",
                cost_per_unit_usd=DEFAULT_INPUT_COST_PER_TOKEN_USD,
                input_cost_per_unit_usd=DEFAULT_INPUT_COST_PER_TOKEN_USD,
                output_cost_per_unit_usd=DEFAULT_OUTPUT_COST_PER_TOKEN_USD,
                description=(
                    "Default fallback pricing for any (provider, model) pair "
                    "without an explicit rule: $0.039 / 1M input tokens, "
                    "$0.18 / 1M output tokens."
                ),
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            logger.info(
                "[DB] Seeded default cost rule: input $%g / token, output $%g / token.",
                DEFAULT_INPUT_COST_PER_TOKEN_USD,
                DEFAULT_OUTPUT_COST_PER_TOKEN_USD,
            )
        return row


def ensure_local_user() -> UserRow:
    """Create (or return) the built-in *local* user.

    Used when OAuth is not configured so the UI always has a real
    account to attach projects, stars and activity to.  This user is
    *not* the example user (which owns the seeded read-only fixtures).
    """
    with get_session() as s:
        row = s.exec(
            select(UserRow).where(UserRow.provider_id == LOCAL_USER_PROVIDER_ID)
        ).first()
        if row is None:
            row = UserRow(
                provider="local",
                provider_id=LOCAL_USER_PROVIDER_ID,
                username="local",
                display_name="Local User",
            )
            s.add(row)
            s.commit()
            s.refresh(row)
        return row


# =====================================================================
# User CRUD
# =====================================================================


def upsert_user(
    provider: str,
    provider_id: str,
    username: str,
    email: str | None = None,
    avatar_url: str | None = None,
    display_name: str | None = None,
) -> UserRow:
    with get_session() as s:
        row = s.exec(select(UserRow).where(UserRow.provider_id == provider_id)).first()
        if row is None:
            row = UserRow(
                provider=provider,
                provider_id=provider_id,
                username=username,
                email=email,
                avatar_url=avatar_url,
                display_name=display_name,
            )
            s.add(row)
        else:
            row.username = username
            row.email = email
            row.avatar_url = avatar_url
            if display_name:
                row.display_name = display_name
            row.last_login_at = datetime.now(timezone.utc)
        s.commit()
        s.refresh(row)
        return row


def get_user(user_id: int) -> Optional[UserRow]:
    with get_session() as s:
        return s.get(UserRow, user_id)


def update_user_profile(
    user_id: int,
    *,
    display_name: str | None = None,
    bio: str | None = None,
    preferences_json: str | None = None,
) -> Optional[UserRow]:
    with get_session() as s:
        row = s.get(UserRow, user_id)
        if row is None:
            return None
        if display_name is not None:
            row.display_name = display_name
        if bio is not None:
            row.bio = bio
        if preferences_json is not None:
            row.preferences_json = preferences_json
        s.commit()
        s.refresh(row)
        return row


# =====================================================================
# Project CRUD
# =====================================================================


def create_project(
    name: str,
    owner_id: int | None = None,
    label: str | None = None,
    raw_text: str | None = None,
    description: str | None = None,
    is_public: bool = False,
) -> ProjectRow:
    with get_session() as s:
        row = ProjectRow(
            name=name,
            owner_id=owner_id,
            label=label,
            raw_text=raw_text,
            description=description,
            is_public=is_public,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row


def get_project(project_id: int) -> Optional[ProjectRow]:
    with get_session() as s:
        return s.get(ProjectRow, project_id)


def find_project_by_name(
    name: str,
    owner_id: int | None = None,
) -> Optional[ProjectRow]:
    """Find a project by name, optionally scoped to an owner."""
    with get_session() as s:
        stmt = select(ProjectRow).where(ProjectRow.name == name)
        if owner_id is not None:
            stmt = stmt.where(ProjectRow.owner_id == owner_id)
        return s.exec(stmt).first()


def list_projects(user_id: int | None = None) -> list[dict]:
    """Return projects visible to *user_id*.

    Includes the user's own projects, projects shared with them, and
    public projects.  Example-user projects are *excluded* — they are
    surfaced separately via :func:`list_example_projects` and copied
    into the user's account on selection.
    """
    example_id = get_example_user_id()

    with get_session() as s:
        stmt = select(ProjectRow)

        if user_id is not None:
            # IDs of projects shared with this user
            shared_ids = [
                m.project_id
                for m in s.exec(
                    select(ProjectMemberRow).where(ProjectMemberRow.user_id == user_id)
                ).all()
            ]
            conditions = [
                ProjectRow.owner_id == user_id,
                ProjectRow.owner_id.is_(None),
                ProjectRow.is_public.is_(True),
            ]
            if shared_ids:
                conditions.append(ProjectRow.id.in_(shared_ids))
            stmt = stmt.where(or_(*conditions))
        else:
            # AUDIT (post-2026-05-26): when no user context is supplied
            # the listing previously leaked *every* project (private and
            # all). Anonymous / pre-auth callers may only see
            # ownerless seeds and explicitly public projects.
            stmt = stmt.where(
                or_(ProjectRow.owner_id.is_(None),
                    ProjectRow.is_public.is_(True))
            )

        # Always hide example-user projects from the regular listing.
        if example_id is not None:
            stmt = stmt.where(
                or_(ProjectRow.owner_id.is_(None),
                    ProjectRow.owner_id != example_id)
            )

        rows = s.exec(stmt.order_by(ProjectRow.updated_at.desc())).all()

        # Single grouped query for version counts \u2014 avoids N+1 lazy
        # ``len(r.versions)`` hits when many projects are listed.
        from sqlmodel import func
        proj_ids = [r.id for r in rows]
        version_counts: dict[int, int] = {}
        if proj_ids:
            count_rows = s.exec(
                select(VersionRow.project_id, func.count(VersionRow.id))
                .where(VersionRow.project_id.in_(proj_ids))
                .group_by(VersionRow.project_id)
            ).all()
            version_counts = {pid: cnt for pid, cnt in count_rows}

        return [
            {
                "id": r.id,
                "name": r.name,
                "label": r.label,
                "description": r.description,
                "owner_id": r.owner_id,
                "is_example": (r.owner_id == example_id) if example_id else False,
                "is_public": r.is_public,
                "forked_from_id": r.forked_from_id,
                "star_count": r.star_count,
                "updated_at": str(r.updated_at),
                "version_count": version_counts.get(r.id, 0),
            }
            for r in rows
        ]


def list_example_projects() -> list[dict]:
    """Return all projects owned by the built-in example user.

    These are the seeded pre-built world models offered on the
    dashboard.  They are intentionally excluded from
    :func:`list_projects`.
    """
    example_id = get_example_user_id()
    if example_id is None:
        return []
    with get_session() as s:
        rows = s.exec(
            select(ProjectRow)
            .where(ProjectRow.owner_id == example_id)
            .order_by(ProjectRow.name)
        ).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "owner_id": r.owner_id,
                "is_example": True,
            }
            for r in rows
        ]


# =====================================================================
# Project management (update, fork)
# =====================================================================


# ---------------------------------------------------------------------
# Round-19 authorization helpers (R19-H1..H5)
# ---------------------------------------------------------------------
#
# Defense-in-depth: prior to R19, the project mutation helpers
# (``update_project``, ``fork_project``, ``add/remove_project_member``,
# ``save_version``, ``promote_branch``) trusted their callers to have
# performed authorization. The UI and MCP server did so, but any
# non-UI/MCP code path (admin script, future helper, third-party
# embedding) could mutate or read across users.
#
# These helpers now accept an optional ``actor_id``. When supplied,
# the helper enforces the matching policy and raises
# :class:`PermissionError` on failure. When ``actor_id`` is ``None``
# the check is skipped and a warning is logged so the unchecked
# path is observable in operational logs. New code paths SHOULD
# always pass ``actor_id``.

# Role precedence for project membership.
_ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2}


def _auth_required() -> bool:
    """True when the deployment runs in hosted (multi-user) mode.

    In hosted mode an ``actor_id is None`` authorization call is a
    programming error (every external surface authenticates a user), so
    the authorization helpers fail closed instead of skipping. Local /
    single-user mode keeps the lenient legacy behaviour.
    """
    try:
        from shadow_loom.settings import get_settings
        return bool(get_settings().oauth.auth_required)
    except Exception:
        return False


def _resolve_project_role(
    s: "Session", project_id: int, actor_id: int
) -> str | None:
    """Return the effective role of *actor_id* on *project_id*.

    Owners implicitly have ``"admin"``; explicit ProjectMemberRow rows
    yield ``viewer | editor | admin``; non-members yield ``None``.
    """
    proj = s.get(ProjectRow, project_id)
    if proj is None:
        return None
    if proj.owner_id == actor_id:
        return "admin"
    member = s.exec(
        select(ProjectMemberRow).where(
            ProjectMemberRow.project_id == project_id,
            ProjectMemberRow.user_id == actor_id,
        )
    ).first()
    return member.role if member else None


def _authorize_project_action(
    s: "Session",
    project_id: int,
    actor_id: int | None,
    *,
    min_role: str,
    operation: str,
) -> None:
    """Raise :class:`PermissionError` if *actor_id* lacks *min_role*.

    When ``actor_id`` is ``None`` the check is skipped and a warning
    is logged so unchecked legacy call sites are observable. Pass
    ``actor_id`` explicitly from any UI / MCP / external caller.
    """
    if actor_id is None:
        if _auth_required():
            raise PermissionError(
                f"{operation}: actor_id is required in hosted mode "
                f"(project {project_id})."
            )
        logger.warning(
            "[db.%s] called without actor_id on project %s — "
            "authorization skipped (legacy call site).",
            operation, project_id,
        )
        return
    role = _resolve_project_role(s, project_id, actor_id)
    if role is None:
        raise PermissionError(
            f"{operation}: user {actor_id} is not a member of project "
            f"{project_id}."
        )
    if _ROLE_RANK.get(role, -1) < _ROLE_RANK.get(min_role, 99):
        raise PermissionError(
            f"{operation}: user {actor_id} has role '{role}' on project "
            f"{project_id}, requires '{min_role}' or higher."
        )


def _authorize_project_read(
    s: "Session",
    project_id: int,
    actor_id: int | None,
    *,
    operation: str,
) -> None:
    """Raise :class:`PermissionError` if *actor_id* cannot read project.

    Read access is granted to owners, any member, and (when the
    project's ``is_public`` flag is set) any authenticated actor.
    Skips the check with a warning when ``actor_id`` is ``None``.
    """
    if actor_id is None:
        if _auth_required():
            raise PermissionError(
                f"{operation}: actor_id is required in hosted mode "
                f"(project {project_id})."
            )
        logger.warning(
            "[db.%s] called without actor_id on project %s — "
            "read authorization skipped (legacy call site).",
            operation, project_id,
        )
        return
    proj = s.get(ProjectRow, project_id)
    if proj is None:
        raise PermissionError(
            f"{operation}: project {project_id} not found."
        )
    if proj.is_public:
        return
    if proj.owner_id == actor_id:
        return
    # Seeded example projects are owned by the built-in example user and
    # are intentionally readable by every authenticated actor so they
    # can be forked from the dashboard. They remain private for write
    # operations because owner/membership checks still apply elsewhere.
    example_user_id = get_example_user_id()
    if example_user_id is not None and proj.owner_id == example_user_id:
        return
    member = s.exec(
        select(ProjectMemberRow).where(
            ProjectMemberRow.project_id == project_id,
            ProjectMemberRow.user_id == actor_id,
        )
    ).first()
    if member is None:
        raise PermissionError(
            f"{operation}: user {actor_id} cannot read private project "
            f"{project_id}."
        )


def update_project(
    project_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    label: str | None = None,
    is_public: bool | None = None,
    raw_text: str | None = None,
    actor_id: int | None = None,
) -> Optional[ProjectRow]:
    with get_session() as s:
        # R19-H1: admin-level membership (owner or admin role) required.
        _authorize_project_action(
            s, project_id, actor_id,
            min_role="admin", operation="update_project",
        )
        row = s.get(ProjectRow, project_id)
        if row is None:
            return None
        if name is not None:
            row.name = name
        if description is not None:
            row.description = description
        if label is not None:
            row.label = label
        if is_public is not None:
            row.is_public = is_public
        if raw_text is not None:
            row.raw_text = raw_text
        row.updated_at = datetime.now(timezone.utc)
        s.commit()
        s.refresh(row)
        return row


def fork_project(
    source_project_id: int,
    new_owner_id: int,
    new_name: str | None = None,
    actor_id: int | None = None,
) -> Optional[ProjectRow]:
    """Fork a project: copy latest version to a new project owned by new_owner_id.

    R19-H3: ``actor_id`` (defaults to ``new_owner_id``) must be allowed
    to read the source project (owner, member, or source ``is_public``).
    R19-H5: the forked v0 is now persisted through :func:`save_version`
    so the same strict WorldStateV1 validation that protects new
    versions also protects forks.
    """
    # Default the actor to the fork target if not explicitly supplied,
    # matching the prior UI/MCP semantics where the fork action is
    # always taken on behalf of the new owner.
    effective_actor = actor_id if actor_id is not None else new_owner_id
    with get_session() as s:
        _authorize_project_read(
            s, source_project_id, effective_actor,
            operation="fork_project",
        )
        source = s.get(ProjectRow, source_project_id)
        if source is None:
            return None
        latest = s.exec(
            select(VersionRow)
            .where(VersionRow.project_id == source_project_id)
            .order_by(VersionRow.version.desc())
        ).first()
        if latest is None:
            return None

        # R19-H5: validate the source payload *before* creating the
        # project shell. The fork carries the source world_state
        # verbatim through save_version below, but the model may have
        # drifted since the source was persisted. Catching that here
        # avoids leaving an orphaned v-less ProjectRow that the
        # post-failure cleanup would otherwise have to mop up.
        _assert_world_state_persistable(latest.world_state_json)

        forked = ProjectRow(
            name=new_name or f"{source.name} (fork)",
            owner_id=new_owner_id,
            label=source.label,
            description=source.description,
            raw_text=source.raw_text,
            forked_from_id=source_project_id,
        )
        s.add(forked)
        s.commit()
        s.refresh(forked)
        forked_id = forked.id
        # Capture source attributes before leaving the session — once
        # ``save_version`` opens its own session below the detached
        # ``latest`` row would lazy-load against a closed session.
        src_world_state_json = latest.world_state_json
        src_world_id = latest.world_id
        src_branch_label = latest.branch_label
        src_prose = latest.prose
        src_raw_query = latest.raw_query
        src_parsed_query_json = latest.parsed_query_json
        src_changeset_json = latest.changeset_json

    # R19-H5: route v0 creation through save_version so the strict
    # WorldStateV1 validation at the persistence boundary protects
    # forks just as it protects ordinary mutations. The fork carries
    # the source's branch identity and provenance verbatim (R11-04).
    try:
        save_version(
            forked_id,
            src_world_state_json,
            version=0,
            source="fork",
            description=f"Forked from project {source_project_id}",
            world_id=src_world_id,
            branch_label=src_branch_label,
            prose=src_prose,
            raw_query=src_raw_query,
            parsed_query_json=src_parsed_query_json,
            changeset_json=src_changeset_json,
            user_id=new_owner_id,
        )
    except Exception:
        # If validation refuses the fork, roll back the orphaned
        # ProjectRow so we don't leave a v-less shell behind.
        with get_session() as s_cleanup:
            stale = s_cleanup.get(ProjectRow, forked_id)
            if stale is not None:
                s_cleanup.delete(stale)
                s_cleanup.commit()
        raise

    with get_session() as s_final:
        return s_final.get(ProjectRow, forked_id)


class ProjectDeleteError(Exception):
    """Raised when a project cannot be deleted."""


def delete_project(project_id: int, user_id: int) -> bool:
    """Hard-delete a project and all dependent rows.

    Owner-only. Refuses to delete projects owned by the built-in
    *example* user so the seeded fixtures cannot be removed via the UI.

    Cascades: VersionRow, ActivityRow, ProjectStarRow, ProjectMemberRow.

    Returns ``True`` on success. Raises ``ProjectDeleteError`` if the
    project does not exist, the caller is not the owner, or the project
    is an example fixture. Raises ``PermissionError`` for non-owners
    so the UI can distinguish forbidden vs. missing cleanly.
    """
    example_user_id = get_example_user_id()
    with get_session() as s:
        proj = s.get(ProjectRow, project_id)
        if proj is None:
            raise ProjectDeleteError(f"Project {project_id} not found")
        if example_user_id is not None and proj.owner_id == example_user_id:
            raise ProjectDeleteError("Example projects cannot be deleted")
        if proj.owner_id != user_id:
            raise PermissionError("Only the project owner can delete it")

        # Delete order matters now that SQLite FK enforcement is on:
        # rows that reference versions.id (ActivityRow.version_id,
        # ActiveVersionRow.version_row_id) must go BEFORE the versions
        # themselves, and versions must be deleted deepest-first so
        # the self-FK ancestor_id never points at a row already gone.
        for act in s.exec(
            select(ActivityRow).where(ActivityRow.project_id == project_id)
        ).all():
            s.delete(act)
        for av in s.exec(
            select(ActiveVersionRow).where(ActiveVersionRow.project_id == project_id)
        ).all():
            s.delete(av)
        for star in s.exec(
            select(ProjectStarRow).where(ProjectStarRow.project_id == project_id)
        ).all():
            s.delete(star)
        for mem in s.exec(
            select(ProjectMemberRow).where(ProjectMemberRow.project_id == project_id)
        ).all():
            s.delete(mem)
        # Dependent rows that carry an FK to projects.id / versions.id must
        # be removed before the versions and project row, otherwise the
        # final DELETEs raise FOREIGN KEY constraint failed under SQLite
        # ``PRAGMA foreign_keys=ON`` (every project that ran the pipeline
        # has agent/api call logs, and most have settings/usage rows).
        for dep_model in (
            WorldFactRow,
            ProjectSettingsRow,
            AgentCallLogRow,
            ApiCallLogRow,
            ProjectUsageSummaryRow,
            McpIdempotencyRow,
        ):
            for dep in s.exec(
                select(dep_model).where(dep_model.project_id == project_id)
            ).all():
                s.delete(dep)
        s.flush()
        # Versions: deepest-first so each delete sees no descendant FK
        # still pointing at it via ancestor_id.
        versions = s.exec(
            select(VersionRow).where(VersionRow.project_id == project_id)
        ).all()
        depth: dict[int, int] = {}

        def _depth(v: VersionRow) -> int:
            if v.id in depth:
                return depth[v.id]
            d = 0
            cur = v
            seen: set[int] = set()
            while cur.ancestor_id is not None and cur.ancestor_id not in seen:
                seen.add(cur.ancestor_id)
                parent = next(
                    (x for x in versions if x.id == cur.ancestor_id), None,
                )
                if parent is None:
                    break
                d += 1
                cur = parent
            depth[v.id] = d
            return d

        for ver in sorted(versions, key=_depth, reverse=True):
            s.delete(ver)
            s.flush()
        # Detach any forks that pointed at this project so the FK doesn't dangle.
        for child in s.exec(
            select(ProjectRow).where(ProjectRow.forked_from_id == project_id)
        ).all():
            child.forked_from_id = None
            s.add(child)
        s.delete(proj)
        s.commit()
        return True


# =====================================================================
# Project Members (collaboration)
# =====================================================================


def add_project_member(
    project_id: int,
    user_id: int,
    role: str = "viewer",
    actor_id: int | None = None,
) -> ProjectMemberRow:
    with get_session() as s:
        # R19-H2: only owners / admin members may modify membership.
        _authorize_project_action(
            s, project_id, actor_id,
            min_role="admin", operation="add_project_member",
        )
        existing = s.exec(
            select(ProjectMemberRow).where(
                ProjectMemberRow.project_id == project_id,
                ProjectMemberRow.user_id == user_id,
            )
        ).first()
        if existing:
            existing.role = role
            s.commit()
            s.refresh(existing)
            return existing
        row = ProjectMemberRow(project_id=project_id, user_id=user_id, role=role)
        s.add(row)
        try:
            s.commit()
        except IntegrityError:
            # Concurrent add of the same (project_id, user_id) member.
            # Reconcile to the row the other writer created and apply the
            # requested role rather than crashing on the unique key.
            s.rollback()
            row = s.exec(
                select(ProjectMemberRow).where(
                    ProjectMemberRow.project_id == project_id,
                    ProjectMemberRow.user_id == user_id,
                )
            ).first()
            if row is None:
                raise
            row.role = role
            s.commit()
        s.refresh(row)
        return row


def remove_project_member(
    project_id: int,
    user_id: int,
    actor_id: int | None = None,
) -> bool:
    with get_session() as s:
        # R19-H2: only owners / admin members may modify membership.
        _authorize_project_action(
            s, project_id, actor_id,
            min_role="admin", operation="remove_project_member",
        )
        row = s.exec(
            select(ProjectMemberRow).where(
                ProjectMemberRow.project_id == project_id,
                ProjectMemberRow.user_id == user_id,
            )
        ).first()
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


def list_project_members(project_id: int) -> list[dict]:
    with get_session() as s:
        rows = s.exec(
            select(ProjectMemberRow).where(ProjectMemberRow.project_id == project_id)
        ).all()
        result = []
        for m in rows:
            user = s.get(UserRow, m.user_id)
            result.append({
                "user_id": m.user_id,
                "username": user.username if user else "unknown",
                "display_name": user.display_name if user else None,
                "avatar_url": user.avatar_url if user else None,
                "role": m.role,
                "created_at": str(m.created_at),
            })
        return result


def get_user_project_role(project_id: int, user_id: int) -> str | None:
    """Return the user's role on a project, or None if not a member.

    Owners implicitly have 'admin' role.
    """
    with get_session() as s:
        proj = s.get(ProjectRow, project_id)
        if proj and proj.owner_id == user_id:
            return "admin"
        member = s.exec(
            select(ProjectMemberRow).where(
                ProjectMemberRow.project_id == project_id,
                ProjectMemberRow.user_id == user_id,
            )
        ).first()
        return member.role if member else None


# =====================================================================
# Project Stars
# =====================================================================


def toggle_star(project_id: int, user_id: int) -> bool:
    """Toggle a star on a project. Returns True if now starred, False if unstarred."""
    from sqlmodel import func

    with get_session() as s:
        existing = s.exec(
            select(ProjectStarRow).where(
                ProjectStarRow.project_id == project_id,
                ProjectStarRow.user_id == user_id,
            )
        ).first()
        if existing:
            s.delete(existing)
            now_starred = False
        else:
            s.add(ProjectStarRow(project_id=project_id, user_id=user_id))
            now_starred = True
        try:
            s.flush()
        except IntegrityError:
            s.rollback()
            concurrent_row = s.exec(
                select(ProjectStarRow).where(
                    ProjectStarRow.project_id == project_id,
                    ProjectStarRow.user_id == user_id,
                )
            ).first()
            if not now_starred or concurrent_row is None:
                # Not the unique-star insert race (e.g. an FK violation on
                # a bogus user/project) — surface the real error instead
                # of silently reporting an incorrect star state.
                raise
            # A concurrent toggle already inserted this star; treat it as
            # starred and just reconcile the counter below.
            now_starred = True
        # Derive the denormalised counter from the authoritative row
        # count so a raced toggle can't double-count or under-count it.
        proj = s.get(ProjectRow, project_id)
        if proj is not None:
            proj.star_count = s.exec(
                select(func.count())
                .select_from(ProjectStarRow)
                .where(ProjectStarRow.project_id == project_id)
            ).one()
        s.commit()
        return now_starred


def is_starred(project_id: int, user_id: int) -> bool:
    with get_session() as s:
        return s.exec(
            select(ProjectStarRow).where(
                ProjectStarRow.project_id == project_id,
                ProjectStarRow.user_id == user_id,
            )
        ).first() is not None


def list_starred_projects(user_id: int) -> list[int]:
    """Return project IDs starred by this user."""
    with get_session() as s:
        rows = s.exec(
            select(ProjectStarRow).where(ProjectStarRow.user_id == user_id)
        ).all()
        return [r.project_id for r in rows]


# =====================================================================
# API Keys
# =====================================================================


def _api_key_pepper() -> bytes:
    """Server-side HMAC pepper for API key hashing.

    AUDIT (post-2026-05-26): plain SHA-256 over a high-entropy bearer
    token is reasonable, but adding a server-side pepper means an
    offline attacker who exfiltrates the ``api_keys`` table still
    cannot brute-force matching candidate tokens without also
    compromising the application secret. The pepper is read from
    ``SHADOW_LOOM_API_KEY_PEPPER`` (preferred) or ``SECRET_KEY`` and
    falls back to an empty value for legacy compatibility on dev
    installs that have not yet provisioned one.
    """
    pep = os.environ.get("SHADOW_LOOM_API_KEY_PEPPER") or os.environ.get("SECRET_KEY") or ""
    return pep.encode("utf-8")


def _hash_api_key(raw_key: str) -> str:
    """Canonical API key hash: HMAC-SHA256 with server-side pepper.

    Bearer tokens are ``sl_<32-byte token_urlsafe>`` — ~256 bits of
    entropy — so a fast keyed hash (HMAC-SHA256) is preferred over a
    memory-hard KDF: Argon2id on every authenticated request would add
    >100ms latency for no meaningful brute-force resistance against a
    256-bit secret. The pepper guards against offline attack on a
    stolen DB snapshot.
    """
    pep = _api_key_pepper()
    if not pep:
        # Legacy / unconfigured installs: preserve previous behaviour
        # so existing stored hashes still validate. Operators should
        # set ``SHADOW_LOOM_API_KEY_PEPPER`` and rotate keys.
        return hashlib.sha256(raw_key.encode()).hexdigest()
    return hmac.new(pep, raw_key.encode("utf-8"), hashlib.sha256).hexdigest()


def _legacy_hash_api_key(raw_key: str) -> str:
    """Pre-pepper SHA-256 hash, kept for backward-compat key lookup."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def create_api_key(
    user_id: int,
    name: str,
    scopes: str = "read,write",
    expires_at: datetime | None = None,
) -> tuple[ApiKeyRow, str]:
    """Create a new API key. Returns (row, raw_key).

    The raw key is only returned once — callers must show it to the user
    immediately. Only the hash is stored.
    """
    raw_key = f"sl_{secrets.token_urlsafe(32)}"
    key_hash = _hash_api_key(raw_key)
    key_prefix = raw_key[:12]

    with get_session() as s:
        row = ApiKeyRow(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            scopes=scopes,
            expires_at=expires_at,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row, raw_key


def validate_api_key(raw_key: str) -> Optional[ApiKeyRow]:
    """Validate a bearer token. Returns the ApiKeyRow if valid, None otherwise."""
    candidate_hashes = {_hash_api_key(raw_key)}
    legacy = _legacy_hash_api_key(raw_key)
    if legacy not in candidate_hashes:
        # AUDIT (post-2026-05-26): accept legacy SHA-256 hashes during
        # the migration window so users whose keys predate the pepper
        # rollout don't lose access. Re-issue + rotate on next login.
        candidate_hashes.add(legacy)
    with get_session() as s:
        row = s.exec(
            select(ApiKeyRow).where(
                ApiKeyRow.key_hash.in_(candidate_hashes),
                ApiKeyRow.is_active.is_(True),
            )
        ).first()
        if row is None:
            return None
        if row.expires_at and row.expires_at < datetime.now(timezone.utc):
            return None
        row.last_used_at = datetime.now(timezone.utc)
        s.commit()
        s.refresh(row)
        return row


def list_api_keys(user_id: int) -> list[dict]:
    """List API keys for a user (metadata only, not the key itself)."""
    with get_session() as s:
        rows = s.exec(
            select(ApiKeyRow).where(ApiKeyRow.user_id == user_id)
        ).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "key_prefix": r.key_prefix,
                "scopes": r.scopes,
                "is_active": r.is_active,
                "last_used_at": str(r.last_used_at) if r.last_used_at else None,
                "expires_at": str(r.expires_at) if r.expires_at else None,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]


def revoke_api_key(key_id: int, user_id: int) -> bool:
    """Revoke an API key. Returns True if found and revoked."""
    with get_session() as s:
        row = s.exec(
            select(ApiKeyRow).where(
                ApiKeyRow.id == key_id,
                ApiKeyRow.user_id == user_id,
            )
        ).first()
        if row is None:
            return False
        row.is_active = False
        s.commit()
    # Drop the in-process MCP bearer-token cache so the revoked key
    # stops resolving to a user immediately. Imported lazily to avoid a
    # cycle when shadow_loom is imported without the MCP package.
    try:
        from shadow_loom_mcp.auth import invalidate_token_cache
        invalidate_token_cache(key_id=key_id)
    except ImportError:
        pass
    return True


# =====================================================================
# Activity Feed
# =====================================================================


def log_activity(
    project_id: int,
    action: str,
    *,
    user_id: int | None = None,
    summary: str | None = None,
    detail_json: str | None = None,
    version_id: int | None = None,
) -> ActivityRow:
    with get_session() as s:
        row = ActivityRow(
            project_id=project_id,
            user_id=user_id,
            action=action,
            summary=summary,
            detail_json=detail_json,
            version_id=version_id,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row


def get_project_activity(project_id: int, limit: int = 50) -> list[dict]:
    with get_session() as s:
        rows = s.exec(
            select(ActivityRow)
            .where(ActivityRow.project_id == project_id)
            .order_by(ActivityRow.created_at.desc())
            .limit(limit)
        ).all()
        result = []
        for r in rows:
            user = s.get(UserRow, r.user_id) if r.user_id else None
            result.append({
                "id": r.id,
                "action": r.action,
                "summary": r.summary,
                "user_id": r.user_id,
                "username": user.username if user else None,
                "avatar_url": user.avatar_url if user else None,
                "version_id": r.version_id,
                "created_at": str(r.created_at),
            })
        return result


def get_user_activity(user_id: int, limit: int = 50) -> list[dict]:
    """Get recent activity across all projects for a user."""
    with get_session() as s:
        rows = s.exec(
            select(ActivityRow)
            .where(ActivityRow.user_id == user_id)
            .order_by(ActivityRow.created_at.desc())
            .limit(limit)
        ).all()
        result = []
        for r in rows:
            proj = s.get(ProjectRow, r.project_id)
            result.append({
                "id": r.id,
                "action": r.action,
                "summary": r.summary,
                "project_id": r.project_id,
                "project_name": proj.name if proj else "unknown",
                "version_id": r.version_id,
                "created_at": str(r.created_at),
            })
        return result


# =====================================================================
# Version CRUD
# =====================================================================


def _next_version_number(s: Session, project_id: int) -> int:
    """Return the next monotonic version number for a project."""
    from sqlmodel import func

    result = s.exec(
        select(func.max(VersionRow.version)).where(
            VersionRow.project_id == project_id
        )
    ).first()
    return (result or 0) + 1 if result is not None else 0


# ---------------------------------------------------------------------
# Round-9 C7: MCP idempotency cache helpers
# ---------------------------------------------------------------------

def _mcp_idempotency_hash(
    project_id: int,
    ancestor_id: int | None,
    idempotency_key: str,
) -> str:
    """Stable sha256 over the dedupe key tuple."""
    import hashlib
    payload = f"{int(project_id)}|{int(ancestor_id) if ancestor_id is not None else ''}|{idempotency_key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mcp_idempotency_ttl_days() -> int:
    """Days after which a cached idempotency row is considered stale.

    ``<= 0`` disables expiry/purge (keep rows forever). Overridable via
    ``SHADOW_LOOM_MCP_IDEMPOTENCY_TTL_DAYS``.
    """
    import os
    try:
        return int(os.environ.get("SHADOW_LOOM_MCP_IDEMPOTENCY_TTL_DAYS", "30"))
    except (TypeError, ValueError):
        return 30


def _as_naive_utc(dt: datetime) -> datetime:
    """Drop tzinfo so SQLite (tz-naive) and aware datetimes compare."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def purge_mcp_idempotency(older_than_days: int | None = None) -> int:
    """Delete idempotency rows older than the TTL. Returns rows removed.

    The table accretes one full response-envelope blob per idempotent
    MCP write and is never otherwise pruned, so without this it grows
    unbounded on long-lived deployments. Called best-effort on startup.
    """
    ttl = older_than_days if older_than_days is not None else _mcp_idempotency_ttl_days()
    if ttl <= 0:
        return 0
    cutoff = _as_naive_utc(datetime.now(timezone.utc)) - timedelta(days=ttl)
    with get_session() as s:
        stale = s.exec(
            select(McpIdempotencyRow).where(
                McpIdempotencyRow.created_at.is_not(None),
                McpIdempotencyRow.created_at < cutoff,
            )
        ).all()
        for row in stale:
            s.delete(row)
        s.commit()
        return len(stale)


def get_mcp_idempotent_response(
    project_id: int,
    ancestor_id: int | None,
    idempotency_key: str,
) -> dict | None:
    """Return the previously cached response envelope for this key,
    or ``None`` when no prior request matched. Used by
    ``shadow_loom_mcp.helpers.run_and_save`` to short-circuit
    client-side retries.
    """
    import json
    key_hash = _mcp_idempotency_hash(project_id, ancestor_id, idempotency_key)
    with get_session() as s:
        row = s.get(McpIdempotencyRow, key_hash)
        if row is None:
            return None
        # Expired rows are a miss: don't replay a response old enough that
        # the purge would have removed it (and may be slated for removal).
        ttl = _mcp_idempotency_ttl_days()
        if ttl > 0 and row.created_at is not None:
            cutoff = _as_naive_utc(datetime.now(timezone.utc)) - timedelta(days=ttl)
            if _as_naive_utc(row.created_at) < cutoff:
                return None
        try:
            return json.loads(row.response_json)
        except (TypeError, ValueError):
            # Corrupt cache row — treat as a miss; the caller will
            # rerun the pipeline and overwrite below.
            return None


def save_mcp_idempotent_response(
    project_id: int,
    ancestor_id: int | None,
    idempotency_key: str,
    response: dict,
    version_row_id: int | None = None,
) -> None:
    """Persist a response envelope under the dedupe key. Existing rows
    are overwritten (last-write-wins) so a successful retry replaces
    a stale failure cache.
    """
    import json
    key_hash = _mcp_idempotency_hash(project_id, ancestor_id, idempotency_key)
    try:
        response_json = json.dumps(response, default=str)
    except (TypeError, ValueError):
        logger.exception(
            "[MCP] Failed to serialise response for idempotency cache "
            "(project=%s ancestor=%s); skipping cache write",
            project_id, ancestor_id,
        )
        return
    with get_session() as s:
        existing = s.get(McpIdempotencyRow, key_hash)
        if existing is not None:
            existing.response_json = response_json
            existing.version_row_id = version_row_id
            existing.created_at = datetime.now(timezone.utc)
            s.add(existing)
        else:
            s.add(McpIdempotencyRow(
                key_hash=key_hash,
                project_id=project_id,
                ancestor_id=ancestor_id,
                version_row_id=version_row_id,
                response_json=response_json,
            ))
        s.commit()


def _assert_world_state_persistable(
    world_state_json: str, *, accept_partial: bool = False
) -> None:
    """Validate a ``world_state_json`` payload at the persistence boundary.

    Strict by default (audit R18-20): a payload that fails WorldStateV1
    validation raises ``ValueError`` so a typed field drift cannot
    silently poison project history. Callers persisting a known-partial
    payload opt out with ``accept_partial=True``; the
    ``SHADOW_LOOM_STRICT_PERSIST`` kill-switch set to a falsey value
    downgrades the failure back to a warning for emergency use.
    """
    if accept_partial:
        return
    try:
        from shadow_loom.models import WorldStateV1
        WorldStateV1.model_validate_json(world_state_json)
    except Exception as exc:
        import os as _os
        env_val = _os.environ.get("SHADOW_LOOM_STRICT_PERSIST", "").lower()
        soft = env_val in {"0", "false", "no", "off"}
        msg = (
            f"world_state_json failed WorldStateV1 validation at "
            f"persistence boundary: {exc}"
        )
        if soft:
            logger.warning(msg + " (SHADOW_LOOM_STRICT_PERSIST disabled.)")
        else:
            raise ValueError(
                msg + " (pass accept_partial=True to bypass, or set "
                "SHADOW_LOOM_STRICT_PERSIST=0 to soften.)"
            ) from exc


def save_version(
    project_id: int,
    world_state_json: str,
    *,
    ancestor_id: int | None = None,
    source: str = "pipeline",
    description: str | None = None,
    changeset_json: str | None = None,
    raw_query: str | None = None,
    parsed_query_json: str | None = None,
    prose: str | None = None,
    user_id: int | None = None,
    version: int | None = None,
    label: str | None = None,
    world_id: str = "factual",
    branch_label: str | None = None,
    pipeline_reextraction_failed: bool = False,
    accept_partial: bool = False,
    actor_id: int | None = None,
) -> VersionRow:
    """Persist a new version node in the version tree.

    If *version* is None it is auto-assigned as the next monotonic
    integer for the project.  Auto-assignment is retried up to a few
    times on ``IntegrityError`` to absorb concurrent writers racing on
    the (project_id, version) unique constraint.

    Hard guard: callers that ran an ingestion / re-extraction pipeline
    must pass ``pipeline_reextraction_failed=True`` if the pipeline
    raised, and ``accept_partial=True`` to acknowledge they intend to
    persist the partial state anyway. Without that explicit
    acknowledgement we refuse to save, so a silent ``except Exception:
    save_version(...)`` cannot quietly persist a half-extracted graph.
    """
    if pipeline_reextraction_failed and not accept_partial:
        raise ValueError(
            "save_version refused: caller passed "
            "pipeline_reextraction_failed=True without accept_partial=True. "
            "The pipeline failed; either persist the previous successful "
            "version or pass accept_partial=True to explicitly accept the "
            "partial extraction."
        )
    # R19-H4: when actor_id is supplied, require editor-or-higher
    # membership on the target project. Skipped (with warning) when
    # actor_id is None so legacy pipeline call sites keep working.
    if actor_id is not None:
        with get_session() as _auth_s:
            _authorize_project_action(
                _auth_s, project_id, actor_id,
                min_role="editor", operation="save_version",
            )
    # Cheap shape-check at the persistence boundary so corrupted
    # payloads are flagged early.
    #
    # Audit R18-20: validation failure is now **strict by default**.
    # The previous behaviour logged a warning and continued unless
    # ``SHADOW_LOOM_STRICT_PERSIST=1`` was set, which meant a typed
    # field drift (e.g. enum rename, new required field) silently
    # poisoned the project history. Callers that need to persist a
    # partial / known-invalid payload (recovery shims, fuzz tests)
    # opt out by passing ``accept_partial=True``. The env var stays
    # as a kill-switch in the opposite direction: setting it to
    # ``"0"`` / ``"false"`` downgrades the failure back to a warning
    # for emergency operational use.
    _assert_world_state_persistable(world_state_json, accept_partial=accept_partial)
    # When an explicit version is supplied we honour it (single attempt).
    max_attempts = 1 if version is not None else 5
    last_err: Exception | None = None
    # Round-6 audit: refuse cross-project ancestry. ``ancestor_id``
    # references a VersionRow row id; without this guard a caller can
    # attach a version of project B to an ancestor in project A,
    # corrupting later ancestry walks (delete cascades, branch
    # promotion, AMWN history) by crossing project boundaries.
    if ancestor_id is not None:
        with get_session() as s_check:
            anc = s_check.get(VersionRow, ancestor_id)
            if anc is None:
                raise ValueError(
                    f"save_version: ancestor_id={ancestor_id} does not "
                    f"exist."
                )
            if anc.project_id != project_id:
                raise ValueError(
                    f"save_version: ancestor_id={ancestor_id} belongs "
                    f"to project {anc.project_id}, not target project "
                    f"{project_id}; refusing to create cross-project "
                    f"ancestry."
                )
    for attempt in range(max_attempts):
        with get_session() as s:
            assigned_version = (
                version if version is not None else _next_version_number(s, project_id)
            )

            row = VersionRow(
                project_id=project_id,
                version=assigned_version,
                ancestor_id=ancestor_id,
                source=source,
                description=description,
                label=label,
                world_state_json=world_state_json,
                changeset_json=changeset_json,
                raw_query=raw_query,
                parsed_query_json=parsed_query_json,
                prose=prose,
                user_id=user_id,
                world_id=world_id,
                branch_label=branch_label,
            )
            s.add(row)

            # Touch project updated_at
            proj = s.get(ProjectRow, project_id)
            if proj:
                proj.updated_at = datetime.now(timezone.utc)

            try:
                s.commit()
            except IntegrityError as e:
                s.rollback()
                # Round-5 audit: the retry loop must only swallow
                # version-uniqueness collisions (two writers racing for
                # the same v-number). Other IntegrityErrors — e.g.
                # ancestor_id FK violations, NOT NULL breaches, the
                # cross-project ancestry guard rewriting itself — are
                # programmer/data errors that must surface immediately,
                # not get masked by N retries that re-trigger the same
                # constraint and finally raise a generic RuntimeError
                # with the original cause buried.
                _msg = (str(getattr(e, "orig", e)) + " " + str(e)).lower()
                _is_version_collision = (
                    "unique" in _msg or "duplicate" in _msg
                ) and "version" in _msg
                if not _is_version_collision:
                    logger.error(
                        "[db.save_version] Non-retryable IntegrityError on "
                        "project %s v%s: %s",
                        project_id, assigned_version, e,
                    )
                    raise
                last_err = e
                logger.warning(
                    "[db.save_version] IntegrityError on project %s v%s (attempt %d/%d) \u2014 retrying with fresh version.",
                    project_id, assigned_version, attempt + 1, max_attempts,
                )
                continue
            s.refresh(row)
            return row

    # Exhausted retries
    raise RuntimeError(
        f"save_version failed after {max_attempts} attempts for project {project_id}"
    ) from last_err


def get_version(project_id: int, version: int) -> Optional[VersionRow]:
    with get_session() as s:
        return s.exec(
            select(VersionRow).where(
                VersionRow.project_id == project_id,
                VersionRow.version == version,
            )
        ).first()


def get_version_by_id(version_row_id: int) -> Optional[VersionRow]:
    with get_session() as s:
        return s.get(VersionRow, version_row_id)


def get_latest_version(project_id: int) -> Optional[VersionRow]:
    with get_session() as s:
        return s.exec(
            select(VersionRow)
            .where(VersionRow.project_id == project_id)
            .order_by(VersionRow.version.desc())
        ).first()


def list_versions(project_id: int) -> list[dict]:
    """Return all versions for a project (flat list, tree info included)."""
    with get_session() as s:
        rows = s.exec(
            select(VersionRow)
            .where(VersionRow.project_id == project_id)
            .order_by(VersionRow.version.asc())
        ).all()
        return [
            {
                "id": r.id,
                "version": r.version,
                "ancestor_id": r.ancestor_id,
                "source": r.source,
                "description": r.description,
                "label": r.label,
                "is_bookmarked": r.is_bookmarked,
                "has_prose": r.prose is not None,
                "has_changeset": r.changeset_json is not None,
                "user_id": r.user_id,
                "created_at": str(r.created_at),
                "world_id": r.world_id,
                "branch_label": r.branch_label,
            }
            for r in rows
        ]


def get_version_tree(project_id: int) -> list[dict]:
    """Return the version tree structure with changeset summaries."""
    with get_session() as s:
        rows = s.exec(
            select(VersionRow)
            .where(VersionRow.project_id == project_id)
            .order_by(VersionRow.version.asc())
        ).all()
        result = []
        for r in rows:
            changeset_summary = None
            if r.changeset_json:
                try:
                    cs = json.loads(r.changeset_json)
                    # Pass through every MergeChangeset counter so UI
                    # surfaces (version sidebar, story-card badges,
                    # audit log, dialogs) can render the full
                    # additive + deletion + affect + supersession
                    # picture without re-reading the raw changeset.
                    _counter_keys = (
                        # Additive
                        "events_added", "causal_edges_added", "spatial_edges_added",
                        "social_edges_added", "information_edges_added",
                        "entity_updates_applied",
                        "entities_added", "objects_added", "locations_added",
                        "world_traits_added",
                        # Affect / belief
                        "propositions_added", "proposition_truths_committed",
                        "proposition_snapshots_added", "concerns_added",
                        "concern_snapshots_added",
                        "belief_confidence_updates_applied",
                        # Deletion
                        "events_removed", "causal_edges_removed",
                        "spatial_edges_removed", "social_edges_removed",
                        "channels_removed", "entities_removed",
                        "objects_removed", "locations_removed",
                        "world_traits_removed", "propositions_removed",
                        "concerns_removed",
                        # Supersession
                        "events_superseded",
                    )
                    changeset_summary = {
                        k: cs.get(k, 0) for k in _counter_keys
                    }
                    # Preserve the skipped-updates list verbatim — useful
                    # for the audit tab to surface entity-update misses.
                    if cs.get("entity_updates_skipped"):
                        changeset_summary["entity_updates_skipped"] = cs.get(
                            "entity_updates_skipped"
                        )
                    # Referential-integrity: list of events whose
                    # actor_ids/target_ids point at unknown ids. The
                    # UI renders this as a warning chip / drawer so
                    # users can promote dangling refs into a follow-up
                    # query.introduce.
                    if cs.get("events_with_dangling_refs"):
                        changeset_summary["events_with_dangling_refs"] = cs.get(
                            "events_with_dangling_refs"
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(
                {
                    "id": r.id,
                    "version": r.version,
                    "ancestor_id": r.ancestor_id,
                    "source": r.source,
                    "description": r.description,
                    "label": r.label,
                    "is_bookmarked": r.is_bookmarked,
                    "changeset_summary": changeset_summary,
                    "user_id": r.user_id,
                    "created_at": str(r.created_at),
                    "world_id": r.world_id,
                    "branch_label": r.branch_label,
                }
            )
        return result


def get_version_lineage(project_id: int, version: int) -> list[dict]:
    """Walk the ancestor chain from a version back to the root (v0).

    Returns an ordered list from root → target version.
    """
    with get_session() as s:
        # Build a lookup of all versions by row id
        rows = s.exec(
            select(VersionRow).where(VersionRow.project_id == project_id)
        ).all()
        by_id: dict[int, VersionRow] = {r.id: r for r in rows}
        by_version: dict[int, VersionRow] = {r.version: r for r in rows}

        target = by_version.get(version)
        if target is None:
            return []

        chain: list[dict] = []
        current: Optional[VersionRow] = target
        while current is not None:
            chain.append(
                {
                    "id": current.id,
                    "version": current.version,
                    "ancestor_id": current.ancestor_id,
                    "source": current.source,
                    "description": current.description,
                    "created_at": str(current.created_at),
                }
            )
            if current.ancestor_id is not None:
                current = by_id.get(current.ancestor_id)
            else:
                current = None

        chain.reverse()
        return chain


def get_version_lineage_rows(version_row_id: int) -> list[VersionRow]:
    """Walk the ancestor chain from ``version_row_id`` back to the root.

    Returns the actual ``VersionRow`` objects ordered root → target so
    callers can rebuild a full ``VersionedWorldModel.history`` (with
    ``world_id``, ``branch_label``, ``prose`` carried per row) when
    re-hydrating a session from the DB. Unlike :func:`get_version_lineage`
    this returns the SQLModel rows themselves rather than a trimmed
    dict shape, so the caller has access to every persisted field.

    Returns ``[]`` when the row does not exist.
    """
    with get_session() as s:
        target = s.get(VersionRow, version_row_id)
        if target is None:
            return []
        # Eager-load the project's version graph once so we can walk
        # ancestors without N round-trips. Branches in the same project
        # share a small DAG so this is cheap.
        rows = s.exec(
            select(VersionRow).where(
                VersionRow.project_id == target.project_id
            )
        ).all()
        # Detach so the caller can use the returned rows after the
        # session closes (SQLModel objects become unusable otherwise).
        for r in rows:
            s.expunge(r)
        by_id: dict[int, VersionRow] = {r.id: r for r in rows}

    chain: list[VersionRow] = []
    seen: set[int] = set()
    current: Optional[VersionRow] = by_id.get(version_row_id)
    while current is not None:
        if current.id in seen:
            # Defensive: a corrupted DAG with a cycle should not loop.
            break
        seen.add(current.id)
        chain.append(current)
        if current.ancestor_id is None:
            break
        current = by_id.get(current.ancestor_id)
    chain.reverse()
    return chain


def get_version_children(version_row_id: int) -> list[dict]:
    """Return direct child versions branching from a given version row."""
    with get_session() as s:
        rows = s.exec(
            select(VersionRow)
            .where(VersionRow.ancestor_id == version_row_id)
            .order_by(VersionRow.version.asc())
        ).all()
        return [
            {
                "id": r.id,
                "version": r.version,
                "source": r.source,
                "description": r.description,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]


# =====================================================================
# AMWN branches (Story-integration plan, Step 6)
# =====================================================================


def list_branches(project_id: int) -> list[dict]:
    """Return one summary per distinct AMWN branch in the project's DAG.

    A *branch* is a maximal contiguous chain of versions sharing the same
    ``world_id``. The factual mainline is always present (``world_id =
    'factual'``); each shadow fork shows up as a separate branch rooted
    at its first ``world_id == 'shadow'`` version.

    Each summary carries:
      * ``world_id``                — 'factual' or 'shadow'
      * ``branch_label``            — human-readable (None for mainline)
      * ``root_version_row_id``     — id of the first version on the branch
      * ``root_ancestor_id``        — fork point (None for mainline)
      * ``head_version_row_id``     — id of the latest version on the branch
      * ``head_version_number``     — monotonic version number of the head
      * ``version_count``           — number of versions on the branch
    """
    with get_session() as s:
        rows = s.exec(
            select(VersionRow)
            .where(VersionRow.project_id == project_id)
            .order_by(VersionRow.version.asc())
        ).all()
    if not rows:
        return []

    by_id: dict[int, VersionRow] = {r.id: r for r in rows}
    # D7 (thirteenth-pass audit): pre-index children once instead of
    # rescanning the full ``rows`` list inside the per-branch walk.
    # The old O(branches * versions) inner loop became visible on
    # long-running interactive sessions (50+ branches \xd7 thousands of
    # versions on a single project) where ``get_version_branches`` is
    # called on every UI version-tree refresh.
    children_by_anc: dict[int, list[VersionRow]] = {}
    for c in rows:
        if c.ancestor_id is None:
            continue
        children_by_anc.setdefault(c.ancestor_id, []).append(c)
    # A version is a *branch root* when it has no ancestor (mainline v0)
    # or when its ancestor lives on a different world_id.
    branches: list[dict] = []
    for r in rows:
        parent = by_id.get(r.ancestor_id) if r.ancestor_id is not None else None
        is_root = parent is None or parent.world_id != r.world_id
        if not is_root:
            continue
        # Walk the entire same-world_id descendant subtree rooted here to
        # count every version on the branch and find its head. A branch may
        # fan out (e.g. two edits forked from one version, or a shadow fork
        # later re-branched within the same world_id); all same-world_id
        # descendants belong to this one branch, so we count them all rather
        # than stopping at the first fork. The canonical head is the
        # highest-version tip; ``get_version_children`` still exposes the
        # individual forks for navigation.
        head = r
        version_count = 0
        stack = [r]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            version_count += 1
            if node.version > head.version:
                head = node
            for c in children_by_anc.get(node.id, ()):
                if c.world_id == node.world_id and c.id not in seen:
                    stack.append(c)
        branches.append({
            "world_id": r.world_id,
            "branch_label": r.branch_label,
            "root_version_row_id": r.id,
            "root_version_number": r.version,
            "root_ancestor_id": r.ancestor_id,
            "head_version_row_id": head.id,
            "head_version_number": head.version,
            "version_count": version_count,
        })
    return branches


def _retag_shadow_to_factual(world_state_json: str) -> str:
    """Rewrite every ``"world_id": "shadow"`` inside a serialized
    WorldStateV1 to ``"factual"`` before persisting a promoted branch.

    ``promote_branch`` flips the *VersionRow*'s ``world_id`` to
    ``"factual"``, but the world-state JSON it persists carries its
    own per-node tags on entities, events, propositions, beliefs,
    concerns, traits, locations, objects, etc. Downstream consumers
    (prose renderer, auditor side-effects, MCP query routing,
    AMWN graph filters) branch on those node tags rather than the row
    tag, so leaving them as ``"shadow"`` makes a "promoted" version
    still look like a shadow branch everywhere except the version DAG.

    The walk is structural rather than a string substitution because
    user-authored prose, descriptions, and source text may contain the
    literal substring ``"world_id": "shadow"`` and we don't want to
    rewrite those. Any non-string ``world_id`` value is left alone.
    """
    try:
        data = json.loads(world_state_json)
    except (TypeError, ValueError):
        # Defensive: if the snapshot isn't parseable JSON, the
        # downstream save_version call will fail loudly anyway. Return
        # the original payload so the original error surfaces instead
        # of being masked by a JSON re-encode failure.
        return world_state_json

    def _walk(node):
        if isinstance(node, dict):
            wid = node.get("world_id")
            if isinstance(wid, str) and wid == "shadow":
                node["world_id"] = "factual"
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(data)
    return json.dumps(data)


def promote_branch(
    version_row_id: int,
    *,
    user_id: int | None = None,
    description: str | None = None,
    force: bool = False,
    actor_id: int | None = None,
) -> VersionRow:
    """Copy a shadow-branch version onto the factual mainline as a new version.

    Creates a *new* mainline VersionRow whose ``world_state_json`` and
    ``prose`` come verbatim from the shadow source, but whose
    ``ancestor_id`` points at the current factual head and whose
    ``world_id`` is forced to ``'factual'``. The shadow source is left
    untouched so the fork remains browsable.

    .. warning::

       Promotion is **copy-forward**, not three-way merge. The shadow
       snapshot replaces the factual world wholesale. If the factual
       mainline has advanced beyond the shadow's fork point, those
       factual-only changes are silently overwritten by the shadow
       contents.

       To guard against accidental data loss this function refuses
       to promote when divergence is detected (factual head is not
       the same row as the shadow's nearest factual ancestor).
       Pass ``force=True`` to acknowledge the overwrite.

    Raises ``VersionMutationError`` if:
      * the source version does not exist;
      * the source already lives on the factual mainline;
      * factual mainline has diverged past the shadow's fork point
        and ``force=False``.
    """
    with get_session() as s:
        src = s.get(VersionRow, version_row_id)
        if src is None:
            raise VersionMutationError(
                f"Version row {version_row_id} not found."
            )
        # R19-H4: editor-or-higher membership required to promote.
        _authorize_project_action(
            s, src.project_id, actor_id,
            min_role="editor", operation="promote_branch",
        )
        if src.world_id == "factual":
            raise VersionMutationError(
                f"Version {version_row_id} already lives on the factual "
                "mainline; nothing to promote."
            )

        # Find the current factual head for this project.
        factual_head = s.exec(
            select(VersionRow)
            .where(VersionRow.project_id == src.project_id)
            .where(VersionRow.world_id == "factual")
            .order_by(VersionRow.version.desc())
        ).first()
        ancestor_id = factual_head.id if factual_head is not None else None

        # Walk back from the shadow source to its first factual
        # ancestor — that's the fork point the shadow branched from.
        # If the fork point is the current factual head, the canon
        # has not advanced since the fork and a copy-forward is
        # safe. Otherwise the factual mainline has diverged and the
        # promotion would silently overwrite those advances.
        rows = s.exec(
            select(VersionRow).where(
                VersionRow.project_id == src.project_id
            )
        ).all()
        by_id: dict[int, VersionRow] = {r.id: r for r in rows}
        fork_point: Optional[VersionRow] = None
        cursor: Optional[VersionRow] = src
        seen: set[int] = set()
        while cursor is not None and cursor.id not in seen:
            seen.add(cursor.id)
            if cursor.world_id == "factual":
                fork_point = cursor
                break
            if cursor.ancestor_id is None:
                break
            cursor = by_id.get(cursor.ancestor_id)

    diverged = (
        fork_point is not None
        and factual_head is not None
        and fork_point.id != factual_head.id
    )
    if diverged and not force:
        raise VersionMutationError(
            f"Cannot promote shadow v{src.version}: factual mainline has "
            f"advanced from fork point v{fork_point.version} (id="
            f"{fork_point.id}) to v{factual_head.version} (id="
            f"{factual_head.id}). Promoting now would overwrite those "
            "factual changes with the shadow snapshot. Pass force=True "
            "to proceed anyway."
        )

    promoted_desc = description or (
        f"Promoted shadow v{src.version}"
        + (f" ({src.branch_label})" if src.branch_label else "")
        + " to factual mainline"
        + (
            f" (force; overwrote factual v{fork_point.version}→"
            f"v{factual_head.version})"
            if diverged else ""
        )
    )

    # Re-tag every per-node ``world_id == "shadow"`` inside the
    # snapshot to ``"factual"`` before persisting. The VersionRow
    # itself flips to ``world_id="factual"`` below, but the
    # WorldStateV1 JSON carries its own per-node tags on entities,
    # events, propositions, beliefs, concerns, traits, locations,
    # objects, etc. — and downstream consumers (prose renderer,
    # auditor side-effects, MCP query routing, AMWN graph filters)
    # branch on those node tags, not the row tag. Without this rewrite
    # a "promoted" version would still look like a shadow branch to
    # every downstream surface that walks the world state JSON.
    promoted_world_state_json = _retag_shadow_to_factual(
        src.world_state_json,
    )

    # Round-11 R11-01: the factual_head + divergence check above ran
    # inside a session that has now been released. Between that
    # read and the save_version() below, another writer could append
    # a new factual VersionRow — making our ``ancestor_id`` point at
    # a now-stale head and silently overwriting that new factual
    # canon (the very data-loss scenario the divergence guard is
    # designed to prevent). Re-validate inside a fresh session
    # immediately before persisting; abort with a clear
    # VersionMutationError if the head moved. We require operators to
    # re-issue the promotion rather than auto-retrying so they can
    # re-inspect the new factual contents first.
    with get_session() as s2:
        latest_factual = s2.exec(
            select(VersionRow)
            .where(VersionRow.project_id == src.project_id)
            .where(VersionRow.world_id == "factual")
            .order_by(VersionRow.version.desc())
        ).first()
    latest_factual_id = latest_factual.id if latest_factual is not None else None
    expected_ancestor_id = factual_head.id if factual_head is not None else None
    if latest_factual_id != expected_ancestor_id:
        raise VersionMutationError(
            "Promotion aborted: factual mainline advanced concurrently "
            f"(expected head id={expected_ancestor_id!r}, now "
            f"id={latest_factual_id!r}). Re-inspect the new factual "
            "head and re-issue the promotion if you still intend to "
            "overwrite it."
        )

    return save_version(
        project_id=src.project_id,
        world_state_json=promoted_world_state_json,
        ancestor_id=ancestor_id,
        source="promote_branch",
        description=promoted_desc,
        changeset_json=src.changeset_json,
        raw_query=src.raw_query,
        parsed_query_json=src.parsed_query_json,
        prose=src.prose,
        user_id=user_id,
        world_id="factual",
        # Preserve the source branch label as structured provenance
        # so callers can query "which factual versions were promoted
        # from shadow branch X" without substring-matching on the
        # free-text description. ``branch_label`` on factual rows is
        # otherwise unused (the convention is that only shadow forks
        # carry one), so re-using the field is safe.
        branch_label=src.branch_label,
    )


# =====================================================================
# Usage and Cost Dashboard Queries
# =====================================================================


def get_user_usage_summary(
    user_id: int, 
    period_type: str = "monthly",
    limit: int = 12
) -> list[dict]:
    """Get usage summaries for a user across time periods.
    
    Returns recent usage data for dashboard charts and metrics.
    """
    with get_session() as s:
        rows = s.exec(
            select(UserUsageSummaryRow)
            .where(
                UserUsageSummaryRow.user_id == user_id,
                UserUsageSummaryRow.period_type == period_type
            )
            .order_by(UserUsageSummaryRow.period_start.desc())
            .limit(limit)
        ).all()
        
        result = []
        for r in rows:
            agent_breakdown = {}
            provider_breakdown = {}
            
            if r.agent_type_breakdown_json:
                try:
                    agent_breakdown = json.loads(r.agent_type_breakdown_json)
                except json.JSONDecodeError:
                    pass
                    
            if r.provider_breakdown_json:
                try:
                    provider_breakdown = json.loads(r.provider_breakdown_json)
                except json.JSONDecodeError:
                    pass
            
            result.append({
                "period_start": r.period_start.strftime("%Y-%m-%d") if r.period_start else None,
                "period_end": r.period_end.strftime("%Y-%m-%d") if r.period_end else None,
                "period_type": r.period_type,
                "total_agent_calls": r.total_agent_calls,
                "total_agent_tokens": r.total_agent_tokens,
                "total_agent_cost_usd": r.total_agent_cost_usd,
                "total_api_calls": r.total_api_calls, 
                "total_api_cost_usd": r.total_api_cost_usd,
                "total_cost_usd": r.total_agent_cost_usd + r.total_api_cost_usd,
                "agent_breakdown": agent_breakdown,
                "provider_breakdown": provider_breakdown,
                "updated_at": r.updated_at.strftime("%Y-%m-%d %H:%M") if r.updated_at else None,
            })
        
        return result


def get_user_lifetime_usage(user_id: int) -> Optional[dict]:
    """Get lifetime totals for a user."""
    with get_session() as s:
        # Get the most recent lifetime summary or aggregate from current data
        lifetime = s.exec(
            select(UserUsageSummaryRow)
            .where(
                UserUsageSummaryRow.user_id == user_id,
                UserUsageSummaryRow.period_type == "lifetime"
            )
            .order_by(UserUsageSummaryRow.updated_at.desc())
        ).first()
        
        if lifetime:
            return {
                "total_agent_calls": lifetime.total_agent_calls,
                "total_agent_tokens": lifetime.total_agent_tokens,
                "total_agent_cost_usd": lifetime.total_agent_cost_usd,
                "total_api_calls": lifetime.total_api_calls,
                "total_api_cost_usd": lifetime.total_api_cost_usd,
                "total_cost_usd": lifetime.total_agent_cost_usd + lifetime.total_api_cost_usd,
            }
        
        # Fallback: aggregate from raw log tables
        from sqlmodel import func
        agent_stats = s.exec(
            select(
                func.count(AgentCallLogRow.id),
                func.sum(AgentCallLogRow.total_tokens),
                func.sum(AgentCallLogRow.estimated_cost_usd)
            )
            .where(AgentCallLogRow.user_id == user_id)
        ).first()
        
        api_stats = s.exec(
            select(
                func.count(ApiCallLogRow.id),
                func.sum(ApiCallLogRow.estimated_cost_usd)
            )
            .where(ApiCallLogRow.user_id == user_id)
        ).first()
        
        agent_calls = agent_stats[0] if agent_stats else 0
        agent_tokens = agent_stats[1] if agent_stats else 0
        agent_cost = agent_stats[2] if agent_stats else 0.0
        api_calls = api_stats[0] if api_stats else 0
        api_cost = api_stats[1] if api_stats else 0.0
        
        return {
            "total_agent_calls": agent_calls or 0,
            "total_agent_tokens": agent_tokens or 0, 
            "total_agent_cost_usd": agent_cost or 0.0,
            "total_api_calls": api_calls or 0,
            "total_api_cost_usd": api_cost or 0.0,
            "total_cost_usd": (agent_cost or 0.0) + (api_cost or 0.0),
        }


def get_project_usage_summaries(
    user_id: int,
    period_type: str = "monthly",
    limit: int = 10
) -> list[dict]:
    """Get usage summaries for projects owned by or shared with a user."""
    with get_session() as s:
        # Get user's project IDs (owned + shared)
        owned_projects = s.exec(
            select(ProjectRow.id, ProjectRow.name)
            .where(ProjectRow.owner_id == user_id)
        ).all()
        
        shared_projects = s.exec(
            select(ProjectRow.id, ProjectRow.name)
            .join(ProjectMemberRow, ProjectRow.id == ProjectMemberRow.project_id)
            .where(ProjectMemberRow.user_id == user_id)
        ).all()
        
        all_projects = {p.id: p.name for p in owned_projects + shared_projects}
        
        if not all_projects:
            return []
        
        # Get recent usage for these projects
        rows = s.exec(
            select(ProjectUsageSummaryRow)
            .join(ProjectRow, ProjectUsageSummaryRow.project_id == ProjectRow.id)
            .where(
                ProjectUsageSummaryRow.project_id.in_(list(all_projects.keys())),
                ProjectUsageSummaryRow.period_type == period_type
            )
            .order_by(
                ProjectUsageSummaryRow.period_start.desc(),
                ProjectUsageSummaryRow.total_agent_cost_usd.desc()
            )
            .limit(limit)
        ).all()
        
        result = []
        for r in rows:
            project_name = all_projects.get(r.project_id, "Unknown")
            result.append({
                "project_id": r.project_id,
                "project_name": project_name,
                "period_start": r.period_start.strftime("%Y-%m-%d") if r.period_start else None,
                "period_end": r.period_end.strftime("%Y-%m-%d") if r.period_end else None,
                "period_type": r.period_type,
                "total_agent_calls": r.total_agent_calls,
                "total_agent_tokens": r.total_agent_tokens,
                "total_agent_cost_usd": r.total_agent_cost_usd,
                "total_api_calls": r.total_api_calls,
                "total_api_cost_usd": r.total_api_cost_usd,
                "total_cost_usd": r.total_agent_cost_usd + r.total_api_cost_usd,
                "versions_created": r.versions_created,
                "updated_at": r.updated_at.strftime("%Y-%m-%d %H:%M") if r.updated_at else None,
            })
        
        return result


def get_recent_agent_activity(
    user_id: int,
    project_id: Optional[int] = None, 
    limit: int = 20
) -> list[dict]:
    """Get recent agent execution logs for activity feed."""
    with get_session() as s:
        query = select(
            AgentCallLogRow.id,
            AgentCallLogRow.agent_type,
            AgentCallLogRow.agent_name,
            AgentCallLogRow.execution_time_ms,
            AgentCallLogRow.total_tokens,
            AgentCallLogRow.estimated_cost_usd,
            AgentCallLogRow.status,
            AgentCallLogRow.created_at,
            AgentCallLogRow.project_id,
            ProjectRow.name.label("project_name")
        ).select_from(
            AgentCallLogRow.__table__.outerjoin(
                ProjectRow.__table__, 
                AgentCallLogRow.project_id == ProjectRow.id
            )
        ).where(
            AgentCallLogRow.user_id == user_id
        )
        
        if project_id:
            query = query.where(AgentCallLogRow.project_id == project_id)
        
        rows = s.exec(
            query.order_by(AgentCallLogRow.created_at.desc()).limit(limit)
        ).all()
        
        result = []
        for r in rows:
            result.append({
                "id": r.id,
                "agent_type": r.agent_type,
                "agent_name": r.agent_name,
                "execution_time_ms": r.execution_time_ms,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": r.estimated_cost_usd or 0.0,
                "status": r.status,
                "project_id": r.project_id,
                "project_name": r.project_name or "Unknown",
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
            })
        
        return result


# =====================================================================
# Version delete + reparent (rejoin)
# =====================================================================


class VersionMutationError(Exception):
    """Raised when a version delete or reparent cannot be performed."""


def _collect_descendant_ids(s, root_id: int) -> set[int]:
    """BFS over the version tree to collect all descendants (inclusive).

    Restricted to the root's ``project_id`` so a pre-existing
    cross-project ancestor link (legacy data from before the
    save_version cross-project guard) cannot drag rows from another
    project into a cascade delete (round-6 audit).
    """
    root = s.get(VersionRow, root_id)
    if root is None:
        return {root_id}
    root_pid = root.project_id
    seen: set[int] = {root_id}
    frontier = [root_id]
    while frontier:
        children = s.exec(
            select(VersionRow).where(
                VersionRow.ancestor_id.in_(frontier),
                VersionRow.project_id == root_pid,
            )
        ).all()
        new_ids = [c.id for c in children if c.id not in seen]
        if not new_ids:
            break
        seen.update(new_ids)
        frontier = new_ids
    return seen


def delete_version(
    version_row_id: int,
    user_id: int,
    *,
    cascade: bool = False,
) -> dict:
    """Delete a version row.

    Behaviour:
      * The root version (``ancestor_id is NULL`` and ``version == 0``)
        cannot be deleted — that would orphan the entire project.
      * The currently-loaded version may be deleted; callers should
        refresh state afterwards.
      * When ``cascade=False`` (default), descendants of the deleted
        version are *re-parented* to the deleted version's parent so
        the tree stays connected (a "rejoin"). When the deleted version
        was the root the operation is rejected.
      * When ``cascade=True``, all descendants are deleted as well.

    Owner-or-editor only. Raises ``VersionMutationError`` for missing
    rows or root-deletion attempts. Raises ``PermissionError`` for
    callers without write access to the project.

    Returns ``{"deleted": [ids], "reparented": {child_id: new_ancestor_id}}``.
    """
    with get_session() as s:
        row = s.get(VersionRow, version_row_id)
        if row is None:
            raise VersionMutationError(f"Version {version_row_id} not found")

        # Permission check: project owner or editor.
        proj = s.get(ProjectRow, row.project_id)
        if proj is None:
            raise VersionMutationError(
                f"Project {row.project_id} for version {version_row_id} not found"
            )
        if proj.owner_id != user_id:
            member = s.exec(
                select(ProjectMemberRow).where(
                    ProjectMemberRow.project_id == row.project_id,
                    ProjectMemberRow.user_id == user_id,
                )
            ).first()
            if member is None or member.role not in ("editor", "admin"):
                raise PermissionError(
                    "Only the project owner or an editor can delete versions"
                )

        # Disallow root deletion: it is the only invariant anchor of the tree.
        if row.ancestor_id is None:
            raise VersionMutationError(
                "The root version (v0) cannot be deleted"
            )

        if cascade:
            descendants = _collect_descendant_ids(s, row.id)
            # Retarget active-version pointers that target any deleted row
            # onto the parent of the deleted subtree (``row.ancestor_id``)
            # so MCP/UI callers don't silently jump to an unrelated branch.
            retarget = row.ancestor_id
            stale = s.exec(
                select(ActiveVersionRow).where(
                    ActiveVersionRow.version_row_id.in_(descendants)
                )
            ).all()
            for av in stale:
                av.version_row_id = retarget
                av.updated_at = datetime.now(timezone.utc)
            # Null out activity log references to versions about to be
            # deleted (FK is enforced under SQLite ``PRAGMA foreign_keys=ON``).
            stale_acts = s.exec(
                select(ActivityRow).where(
                    ActivityRow.version_id.in_(descendants)
                )
            ).all()
            for act in stale_acts:
                act.version_id = None
            # Same for the call-log / idempotency rows that FK versions.id;
            # these are historical records, so null the version ref rather
            # than delete them, mirroring the activity handling above.
            for log in s.exec(
                select(AgentCallLogRow).where(
                    AgentCallLogRow.version_id.in_(descendants)
                )
            ).all():
                log.version_id = None
            for log in s.exec(
                select(ApiCallLogRow).where(
                    ApiCallLogRow.version_id.in_(descendants)
                )
            ).all():
                log.version_id = None
            for idem in s.exec(
                select(McpIdempotencyRow).where(
                    McpIdempotencyRow.version_row_id.in_(descendants)
                )
            ).all():
                idem.version_row_id = None
            s.flush()
            # Delete in true depth order (leaves first) to satisfy the
            # self-FK on ``ancestor_id`` regardless of row insertion
            # order.
            # AUDIT (post-2026-05-26): the previous ``sorted(..., reverse=True)``
            # used row.id as a proxy for depth, which fails when a
            # later-inserted row was *re-parented* to be an ancestor of
            # earlier-inserted rows (a legal outcome of ``reparent_version``).
            # We now build the parent map from the rows themselves and
            # peel leaves iteratively.
            desc_set = set(descendants)
            rows_by_id = {
                v.id: v
                for v in s.exec(
                    select(VersionRow).where(VersionRow.id.in_(descendants))
                ).all()
            }
            remaining = set(desc_set)
            while remaining:
                # A "leaf" in this subtree has no remaining child in the set.
                children_of: dict[int, set[int]] = {rid: set() for rid in remaining}
                for rid in remaining:
                    anc = rows_by_id[rid].ancestor_id
                    if anc in children_of:
                        children_of[anc].add(rid)
                leaves = [rid for rid, kids in children_of.items() if not kids]
                if not leaves:
                    # Cycle (should be impossible — ``reparent_version``
                    # rejects them) but fail safely rather than spinning.
                    raise VersionMutationError(
                        "Cascade delete encountered a cycle in version ancestry"
                    )
                for rid in leaves:
                    v = rows_by_id.pop(rid)
                    s.delete(v)
                    remaining.remove(rid)
                s.flush()
            proj.updated_at = datetime.now(timezone.utc)
            s.commit()
            return {"deleted": sorted(descendants), "reparented": {}}

        # Rejoin: reparent direct children onto row.ancestor_id.
        new_ancestor = row.ancestor_id
        children = s.exec(
            select(VersionRow).where(VersionRow.ancestor_id == row.id)
        ).all()
        reparented: dict[int, int] = {}
        for child in children:
            child.ancestor_id = new_ancestor
            reparented[child.id] = new_ancestor
        # Flush so the FK update is staged before we delete the parent;
        # without this SQLAlchemy may interleave the parent-delete with
        # the child-update and null out the child FKs.
        s.flush()

        # Retarget active-version pointers that target the row we're about
        # to delete onto the parent (``new_ancestor``) rather than dropping
        # them, so MCP/UI callers don't silently jump to an unrelated
        # latest branch when their active version is removed.
        stale = s.exec(
            select(ActiveVersionRow).where(
                ActiveVersionRow.version_row_id == version_row_id
            )
        ).all()
        for av in stale:
            av.version_row_id = new_ancestor
            av.updated_at = datetime.now(timezone.utc)

        # Null out activity-log references to the version we're deleting
        # (FK enforced under SQLite ``PRAGMA foreign_keys=ON``).
        stale_acts = s.exec(
            select(ActivityRow).where(
                ActivityRow.version_id == version_row_id
            )
        ).all()
        for act in stale_acts:
            act.version_id = None
        # Null the call-log / idempotency rows that FK this version too,
        # else the s.delete(row) below trips the versions.id FK.
        for log in s.exec(
            select(AgentCallLogRow).where(
                AgentCallLogRow.version_id == version_row_id
            )
        ).all():
            log.version_id = None
        for log in s.exec(
            select(ApiCallLogRow).where(
                ApiCallLogRow.version_id == version_row_id
            )
        ).all():
            log.version_id = None
        for idem in s.exec(
            select(McpIdempotencyRow).where(
                McpIdempotencyRow.version_row_id == version_row_id
            )
        ).all():
            idem.version_row_id = None
        s.flush()

        s.delete(row)
        proj.updated_at = datetime.now(timezone.utc)
        s.commit()
        return {"deleted": [version_row_id], "reparented": reparented}


def reparent_version(
    version_row_id: int,
    new_ancestor_id: Optional[int],
    user_id: int,
) -> bool:
    """Move a version under a new ancestor (manual rejoin / branch graft).

    Constraints:
      * Both versions must belong to the same project.
      * ``new_ancestor_id`` must not be a descendant of ``version_row_id``
        (would create a cycle).
      * Cannot reparent the root.
      * The version's own ID is rejected as new ancestor (self-loop).

    Owner-or-editor only.
    """
    with get_session() as s:
        row = s.get(VersionRow, version_row_id)
        if row is None:
            raise VersionMutationError(f"Version {version_row_id} not found")
        if row.ancestor_id is None:
            raise VersionMutationError("Cannot reparent the root version")
        if new_ancestor_id == version_row_id:
            raise VersionMutationError("A version cannot be its own ancestor")

        proj = s.get(ProjectRow, row.project_id)
        if proj is None:
            raise VersionMutationError(
                f"Project {row.project_id} for version {version_row_id} not found"
            )
        if proj.owner_id != user_id:
            member = s.exec(
                select(ProjectMemberRow).where(
                    ProjectMemberRow.project_id == row.project_id,
                    ProjectMemberRow.user_id == user_id,
                )
            ).first()
            if member is None or member.role not in ("editor", "admin"):
                raise PermissionError(
                    "Only the project owner or an editor can reparent versions"
                )

        if new_ancestor_id is None:
            raise VersionMutationError(
                "A non-root version must have an ancestor"
            )

        new_ancestor = s.get(VersionRow, new_ancestor_id)
        if new_ancestor is None or new_ancestor.project_id != row.project_id:
            raise VersionMutationError(
                "New ancestor must belong to the same project"
            )
        descendants = _collect_descendant_ids(s, row.id)
        if new_ancestor_id in descendants:
            raise VersionMutationError(
                "Cannot reparent under a descendant (would create a cycle)"
            )

        row.ancestor_id = new_ancestor_id
        proj.updated_at = datetime.now(timezone.utc)
        s.commit()
        return True


# =====================================================================
# Version bookmarks, labels, prose export
# =====================================================================


def bookmark_version(version_row_id: int, bookmarked: bool = True) -> bool:
    """Toggle bookmark on a version."""
    with get_session() as s:
        row = s.get(VersionRow, version_row_id)
        if row is None:
            return False
        row.is_bookmarked = bookmarked
        s.commit()
        return True


def label_version(version_row_id: int, label: str | None) -> bool:
    """Set or clear a label on a version."""
    with get_session() as s:
        row = s.get(VersionRow, version_row_id)
        if row is None:
            return False
        row.label = label
        s.commit()
        return True


# =====================================================================
# Active version pointer (per-user-per-project)
# =====================================================================


def set_active_version(
    project_id: int,
    user_id: int,
    version_row_id: int,
) -> "ActiveVersionRow":
    """Upsert the active-version pointer for ``(project_id, user_id)``.

    Validates that ``version_row_id`` belongs to ``project_id``. The
    upsert is retried on ``IntegrityError`` to absorb the read-then-
    insert race when multiple callers (UI, MCP, autosave) write the
    pointer concurrently.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        with get_session() as s:
            ver = s.get(VersionRow, version_row_id)
            if ver is None:
                raise ValueError(f"Version {version_row_id} not found")
            if ver.project_id != project_id:
                raise ValueError(
                    f"Version {version_row_id} does not belong to project {project_id}"
                )
            row = s.exec(
                select(ActiveVersionRow).where(
                    ActiveVersionRow.project_id == project_id,
                    ActiveVersionRow.user_id == user_id,
                )
            ).first()
            if row is None:
                row = ActiveVersionRow(
                    project_id=project_id,
                    user_id=user_id,
                    version_row_id=version_row_id,
                )
                s.add(row)
            else:
                row.version_row_id = version_row_id
                row.updated_at = datetime.now(timezone.utc)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                if attempt + 1 >= max_attempts:
                    raise
                continue
            s.refresh(row)
            return row
    # Unreachable: loop either returns or re-raises on the final attempt.
    raise RuntimeError("set_active_version: exhausted retries without resolution")


def get_active_version(
    project_id: int,
    user_id: int,
) -> Optional[VersionRow]:
    """Return the user's active version row for a project, or ``None``.

    Read-only: if the active pointer references a version that has
    since been deleted, ``None`` is returned but the stale pointer is
    *not* mutated. Callers that want to reclaim the row should call
    :func:`clear_active_version` explicitly. Keeping reads side-effect
    free matters for MCP tools that route through this function.
    """
    with get_session() as s:
        row = s.exec(
            select(ActiveVersionRow).where(
                ActiveVersionRow.project_id == project_id,
                ActiveVersionRow.user_id == user_id,
            )
        ).first()
        if row is None:
            return None
        ver = s.get(VersionRow, row.version_row_id)
        if ver is None:
            return None
        return ver


def get_lineage_to_root(version_row_id: int) -> list[int]:
    """Return the ancestor chain for ``version_row_id`` ordered root \u2192 leaf.

    Walks ``VersionRow.ancestor_id`` upward, defends against cycles
    (corrupt data) by short-circuiting on revisits, and returns the
    list in *root-first* order so it can be passed directly as a
    ``branch_path`` argument to :func:`get_all_prose` (which preserves
    the supplied ordering). Returns an empty list when the version row
    is missing.
    """
    chain: list[int] = []
    seen: set[int] = set()
    cur_id: int | None = version_row_id
    with get_session() as s:
        while cur_id is not None and cur_id not in seen:
            seen.add(cur_id)
            row = s.get(VersionRow, cur_id)
            if row is None:
                break
            chain.append(row.id)
            cur_id = row.ancestor_id
    chain.reverse()
    return chain


def clear_active_version(project_id: int, user_id: int) -> bool:
    """Drop the active-version pointer for ``(project_id, user_id)``."""
    with get_session() as s:
        row = s.exec(
            select(ActiveVersionRow).where(
                ActiveVersionRow.project_id == project_id,
                ActiveVersionRow.user_id == user_id,
            )
        ).first()
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


def get_all_prose(
    project_id: int,
    *,
    branch_path: list[int] | None = None,
) -> list[dict]:
    """Return all versions with prose, ordered by version number.

    Useful for exporting the full story.

    When ``branch_path`` is supplied, the result is filtered to *only*
    those version_row_ids (in the order they appear in ``branch_path``).
    This lets MCP clients walk a specific lineage through the AMWN DAG —
    e.g. a shadow fork's prose plus the factual prefix it diverged
    from — instead of getting the implicit linear ``ORDER BY version``
    that mixes branches together (Story-integration plan, Step 6).
    """
    with get_session() as s:
        if branch_path is not None:
            rows = s.exec(
                select(VersionRow)
                .where(
                    VersionRow.project_id == project_id,
                    VersionRow.id.in_(branch_path),
                    VersionRow.prose.isnot(None),
                )
            ).all()
            order = {rid: i for i, rid in enumerate(branch_path)}
            rows = sorted(rows, key=lambda r: order.get(r.id, 1 << 30))
        else:
            rows = s.exec(
                select(VersionRow)
                .where(
                    VersionRow.project_id == project_id,
                    VersionRow.prose.isnot(None),
                )
                .order_by(VersionRow.version.asc())
            ).all()
        return [
            {
                "version": r.version,
                "source": r.source,
                "description": r.description,
                "prose": r.prose,
                "created_at": str(r.created_at),
                "world_id": r.world_id,
                "branch_label": r.branch_label,
                "version_row_id": r.id,
            }
            for r in rows
        ]


# =====================================================================
# User search (for collaboration invites)
# =====================================================================


def search_users(query: str, limit: int = 10) -> list[dict]:
    """Search users by username or email prefix."""
    with get_session() as s:
        rows = s.exec(
            select(UserRow)
            .where(
                or_(
                    UserRow.username.contains(query),
                    UserRow.email.contains(query),
                )
            )
            .where(UserRow.is_example.is_(False))
            .limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "username": r.username,
                "display_name": r.display_name,
                "avatar_url": r.avatar_url,
            }
            for r in rows
        ]


# =====================================================================
# Backward-compatible aliases (used by old UI code)
# =====================================================================


def save_snapshot(
    project_id: int,
    version: int,
    world_state_json: str,
    description: str | None = None,
) -> VersionRow:
    """Backward-compatible wrapper — saves a version with minimal fields."""
    return save_version(
        project_id=project_id,
        world_state_json=world_state_json,
        version=version,
        source="pipeline",
        description=description,
    )


def load_latest_snapshot(project_id: int) -> Optional[VersionRow]:
    """Backward-compatible alias for ``get_latest_version``."""
    return get_latest_version(project_id)


def load_snapshot(project_id: int, version: int) -> Optional[VersionRow]:
    """Backward-compatible alias for ``get_version``."""
    return get_version(project_id, version)


# =====================================================================
# Research cache helpers
# =====================================================================

def get_cached_research(
    *, user_id: Optional[int], key: str,
) -> Optional["ResearchCacheRow"]:
    """Look up a cached research-provider call for *user_id* + *key*."""
    with get_session() as s:
        q = select(ResearchCacheRow).where(ResearchCacheRow.key == key)
        if user_id is None:
            q = q.where(ResearchCacheRow.user_id.is_(None))
        else:
            q = q.where(ResearchCacheRow.user_id == user_id)
        return s.exec(q).first()


def save_cached_research(
    *,
    user_id: Optional[int],
    key: str,
    provider: str,
    provider_model: str,
    query: str,
    snippets_json: str,
) -> None:
    """Insert (or no-op on conflict) a research cache row."""
    with get_session() as s:
        existing = s.exec(
            select(ResearchCacheRow).where(
                ResearchCacheRow.key == key,
                (ResearchCacheRow.user_id == user_id)
                if user_id is not None
                else ResearchCacheRow.user_id.is_(None),
            )
        ).first()
        if existing is not None:
            return  # cache rows are immutable
        s.add(ResearchCacheRow(
            user_id=user_id,
            key=key,
            provider=provider,
            provider_model=provider_model,
            query=query,
            snippets_json=snippets_json,
        ))
        try:
            s.commit()
        except IntegrityError:
            # Concurrent insert from another worker — fine, theirs wins.
            s.rollback()


# =====================================================================
# WorldFact persistence helpers
# =====================================================================

def list_world_facts(project_id: int) -> list["WorldFactRow"]:
    """Return all ``WorldFactRow`` records for *project_id*."""
    with get_session() as s:
        return list(
            s.exec(
                select(WorldFactRow).where(WorldFactRow.project_id == project_id)
            ).all()
        )


def allocate_world_fact_id(project_id: int) -> str:
    """Return the next free ``FACT_NNN`` identifier for *project_id*.

    Round-10 R10-04: the previous call site used
    ``f"FACT_{len(existing) + 1:03d}"`` which (a) raced under
    concurrent research_topic MCP calls — two callers both saw N
    facts and both tried to insert ``FACT_{N+1:03d}``, second one
    crashing on the uq_world_fact_project_id constraint — and (b)
    silently reused ids after any delete (a deleted FACT_005 would
    make the next allocation collide with FACT_004's successor).

    This helper instead scans the existing ``fact_id`` values, parses
    the numeric suffix from any ``FACT_<n>`` shape (ignoring non-
    conforming ids), and returns max(n)+1 zero-padded to width 3.
    The IntegrityError retry loop in callers turns a lost race into a
    transparent re-allocation rather than a 500.
    """
    with get_session() as s:
        rows = s.exec(
            select(WorldFactRow.fact_id).where(WorldFactRow.project_id == project_id)
        ).all()
    highest = 0
    for fid in rows:
        if not isinstance(fid, str) or not fid.startswith("FACT_"):
            continue
        try:
            n = int(fid[len("FACT_"):])
        except ValueError:
            continue
        if n > highest:
            highest = n
    return f"FACT_{highest + 1:03d}"


def upsert_world_fact(
    *,
    project_id: int,
    fact_id: str,
    topic: str,
    summary: str,
    confidence: str,
    source_url_primary: str,
    provider: str,
    related_node_ids_json: str,
    raw_snippets_json: str,
    retrieved_at: Optional[datetime] = None,
) -> "WorldFactRow":
    """Insert or update a ``WorldFactRow`` for *(project_id, fact_id)*."""
    with get_session() as s:
        row = s.exec(
            select(WorldFactRow).where(
                WorldFactRow.project_id == project_id,
                WorldFactRow.fact_id == fact_id,
            )
        ).first()
        now = datetime.now(timezone.utc)
        if row is None:
            row = WorldFactRow(
                project_id=project_id,
                fact_id=fact_id,
                topic=topic,
                summary=summary,
                confidence=confidence,
                source_url_primary=source_url_primary or None,
                provider=provider,
                related_node_ids_json=related_node_ids_json,
                raw_snippets_json=raw_snippets_json,
                retrieved_at=retrieved_at or now,
            )
            s.add(row)
        else:
            row.topic = topic
            row.summary = summary
            row.confidence = confidence
            row.source_url_primary = source_url_primary or None
            row.provider = provider
            row.related_node_ids_json = related_node_ids_json
            row.raw_snippets_json = raw_snippets_json
            if retrieved_at is not None:
                row.retrieved_at = retrieved_at
        s.commit()
        s.refresh(row)
        return row


def delete_world_fact(*, project_id: int, fact_id: str) -> bool:
    """Delete a ``WorldFactRow``. Returns True iff a row was removed."""
    with get_session() as s:
        row = s.exec(
            select(WorldFactRow).where(
                WorldFactRow.project_id == project_id,
                WorldFactRow.fact_id == fact_id,
            )
        ).first()
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


# =====================================================================
# ProjectSettings persistence helpers
# =====================================================================

def get_project_settings(project_id: int) -> dict:
    """Return the project's settings as a plain dict.

    Always returns a dict — for projects that have never had settings
    written, returns ``{"research_topics": []}``. Read-only: the row is
    not auto-created here, only by :func:`set_project_settings`.
    """
    import json as _json

    with get_session() as s:
        row = s.get(ProjectSettingsRow, project_id)
        if row is None:
            return {"research_topics": []}
        try:
            topics = _json.loads(row.research_topics_json or "[]")
        except (ValueError, TypeError):
            topics = []
        if not isinstance(topics, list):
            topics = []
        return {"research_topics": [str(t) for t in topics]}


def set_project_settings(
    project_id: int,
    *,
    research_topics: list[str],
) -> "ProjectSettingsRow":
    """Upsert per-project settings. Returns the persisted row.

    Validates that *research_topics* is a list of strings; raises
    ``ValueError`` otherwise. Topics are stripped + de-duplicated
    (order preserved).
    """
    import json as _json

    if not isinstance(research_topics, list):
        raise ValueError("research_topics must be a list of strings")
    seen: set[str] = set()
    cleaned: list[str] = []
    for t in research_topics:
        if not isinstance(t, str):
            raise ValueError("research_topics must contain only strings")
        s_t = t.strip()
        if not s_t or s_t in seen:
            continue
        seen.add(s_t)
        cleaned.append(s_t)

    with get_session() as s:
        row = s.get(ProjectSettingsRow, project_id)
        if row is None:
            row = ProjectSettingsRow(
                project_id=project_id,
                research_topics_json=_json.dumps(cleaned),
            )
            s.add(row)
        else:
            row.research_topics_json = _json.dumps(cleaned)
            row.updated_at = datetime.now(timezone.utc)
        s.commit()
        s.refresh(row)
        return row


# =====================================================================
# UserModelSettings persistence helpers
# =====================================================================

# Recognised stage keys, mirroring the per-stage Settings classes in
# shadow_loom/settings.py. Unknown keys are silently dropped.
_RECOGNISED_STAGE_KEYS: frozenset[str] = frozenset({
    "generation",
    "auditor",
    "auditor_generation",
    "extraction",
    "query_parsing",
})


# ---------------------------------------------------------------------
# Round-10 R10-03: at-rest encryption for custom-provider API keys.
#
# Custom providers carry secrets (paid OpenAI-compatible bearer
# tokens) that previously sat plaintext in ``custom_providers_json``.
# Anyone with read access to the DB — or to a forgotten backup — got
# the keys. We now Fernet-encrypt the ``api_key`` field per row, keyed
# off ``SHADOW_LOOM_SECRET_KEY`` (must be a 32-byte urlsafe base64
# string; generate with ``python -c "from cryptography.fernet import
# Fernet; print(Fernet.generate_key().decode())"``).
#
# Backwards compatibility: rows persisted before this patch have raw
# plaintext in ``api_key``. ``_decrypt_api_key`` detects the
# ``ENC1:`` prefix and falls back to returning the input unchanged
# when absent, so existing rows decrypt transparently and get
# re-encrypted on the next ``set_user_model_settings`` write.
#
# When the env var is unset, ``_get_fernet`` returns ``None`` and the
# helpers no-op (plaintext storage continues) so a fresh dev clone
# without the secret keeps working — operators are warned once per
# process via ``_warn_missing_secret``.
# ---------------------------------------------------------------------

_ENC_PREFIX = "ENC1:"
_fernet_cached: object | None = None
_fernet_warned: bool = False
# Round-13 R13-05: snapshot of the key the cached Fernet was built
# from. ``_get_fernet`` re-reads ``SHADOW_LOOM_SECRET_KEY`` whenever
# the env value diverges from the snapshot so secret rotation or
# test-time env mutations are picked up without a process restart.
_fernet_cached_key: str | None = None


def reset_fernet_cache() -> None:
    """Drop the memoised Fernet instance.

    Round-13 R13-05: lets tests and operator-driven secret rotation
    invalidate the cache deterministically rather than relying on the
    env-snapshot heuristic in ``_get_fernet``.
    """
    global _fernet_cached, _fernet_cached_key, _fernet_warned
    _fernet_cached = None
    _fernet_cached_key = None
    _fernet_warned = False


class InvalidSecretKeyError(RuntimeError):
    """Raised when ``SHADOW_LOOM_SECRET_KEY`` is set but unusable.

    Round-13 R13-04: previously this condition silently downgraded to
    plaintext storage, which is a fail-open misconfiguration —
    operators thought their provider API keys were encrypted at rest
    when in fact a typo in the env var had disabled encryption. We
    now raise so callers can surface a clear startup error.
    """


def _get_fernet():
    """Return a memoised Fernet instance, or ``None`` when no secret
    is configured. Local import keeps ``cryptography`` an optional
    runtime dep at the module level.

    Round-13 R13-04: when the env var is set but invalid we raise
    :class:`InvalidSecretKeyError` rather than returning ``None``,
    so encryption never silently downgrades to plaintext.
    Round-13 R13-05: the cache is invalidated whenever the env value
    changes so secret rotation takes effect without a restart.
    """
    global _fernet_cached, _fernet_cached_key, _fernet_warned
    import os as _os
    key = _os.environ.get("SHADOW_LOOM_SECRET_KEY", "").strip()
    if _fernet_cached is not None and _fernet_cached_key == key:
        return _fernet_cached
    # Env diverged from snapshot — drop the cache and re-derive.
    _fernet_cached = None
    _fernet_cached_key = None
    if not key:
        if not _fernet_warned:
            logger.warning(
                "[DB] SHADOW_LOOM_SECRET_KEY is unset; custom-provider "
                "API keys will be stored as PLAINTEXT. Set the env var "
                "(value: output of "
                "`python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\"`) to enable "
                "at-rest encryption."
            )
            _fernet_warned = True
        return None
    try:
        from cryptography.fernet import Fernet
        _fernet_cached = Fernet(key.encode("utf-8"))
        _fernet_cached_key = key
    except Exception as exc:
        # Round-13 R13-04: fail closed. Logging the exception alone
        # would let the application keep running with PLAINTEXT
        # encryption silently — exactly the failure mode an operator
        # who set the env var was trying to avoid.
        logger.exception(
            "[DB] SHADOW_LOOM_SECRET_KEY is set but cryptography "
            "rejected it. Refusing to fall back to plaintext storage. "
            "The key must be a 32-byte urlsafe base64 string."
        )
        raise InvalidSecretKeyError(
            "SHADOW_LOOM_SECRET_KEY is set but invalid; refusing to "
            "downgrade to plaintext at-rest storage. Regenerate with "
            "`python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"`."
        ) from exc
    return _fernet_cached


def _encrypt_api_key(plain: str) -> str:
    """Return the Fernet ciphertext for *plain*, prefixed with
    ``ENC1:`` so :func:`_decrypt_api_key` can tell encrypted strings
    apart from legacy plaintext. Returns the input unchanged when no
    secret is configured or the input is empty.
    """
    if not plain:
        return plain
    f = _get_fernet()
    if f is None:
        return plain
    try:
        token = f.encrypt(plain.encode("utf-8")).decode("ascii")
    except Exception as exc:
        # A Fernet instance exists, so the operator expects at-rest
        # encryption. Silently storing plaintext would defeat the
        # R13-04 fail-closed contract and leak a secret indistinguishably
        # from a legacy row. Fail the save instead.
        logger.exception("[DB] Failed to encrypt api_key; refusing plaintext fallback.")
        raise InvalidSecretKeyError(
            "Failed to encrypt api_key for at-rest storage"
        ) from exc
    return _ENC_PREFIX + token


def _decrypt_api_key(stored: str) -> str:
    """Reverse of :func:`_encrypt_api_key`. Plaintext rows (no
    ``ENC1:`` prefix) and empty strings pass through unchanged so
    pre-encryption deployments keep working.
    """
    if not stored or not stored.startswith(_ENC_PREFIX):
        return stored
    f = _get_fernet()
    if f is None:
        # We have an encrypted blob but no key — surface as empty
        # rather than leaking the ciphertext into LLM client init.
        logger.error(
            "[DB] Encountered ENC1-prefixed api_key but "
            "SHADOW_LOOM_SECRET_KEY is unset; returning empty."
        )
        return ""
    try:
        return f.decrypt(stored[len(_ENC_PREFIX):].encode("ascii")).decode("utf-8")
    except Exception:
        logger.exception(
            "[DB] Failed to decrypt api_key (corrupt ciphertext or "
            "wrong secret); returning empty."
        )
        return ""


def _mask_api_key(plain: str) -> str:
    """Return a UI-safe masked rendering of *plain* — only the last 4
    characters are visible. Used wherever a settings surface echoes
    a stored key back to the operator without giving them the full
    secret. Empty input returns empty.
    """
    if not plain:
        return ""
    tail = plain[-4:] if len(plain) >= 4 else plain
    return f"****{tail}"


def get_user_model_settings(user_id: int) -> dict:
    """Return the user's saved model settings as a plain dict.

    Always returns a dict with the same shape — empty string / empty
    list / empty dict members mean "fall back to the env default". A
    user that has never opened the Settings page returns the all-empty
    template.
    """
    import json as _json

    empty: dict = {
        "default_model": "",
        "stage_models": {},
        "custom_providers": [],
    }
    with get_session() as s:
        row = s.get(UserModelSettingsRow, user_id)
        if row is None:
            return empty
        try:
            stage_models = _json.loads(row.stage_models_json or "{}")
        except (ValueError, TypeError):
            stage_models = {}
        try:
            providers = _json.loads(row.custom_providers_json or "[]")
        except (ValueError, TypeError):
            providers = []
        if not isinstance(stage_models, dict):
            stage_models = {}
        if not isinstance(providers, list):
            providers = []
        # Sanitise.
        cleaned_stages = {
            k: str(v).strip()
            for k, v in stage_models.items()
            if k in _RECOGNISED_STAGE_KEYS and isinstance(v, str) and v.strip()
        }
        cleaned_providers: list[dict] = []
        seen: set[str] = set()
        for p in providers:
            if not isinstance(p, dict):
                continue
            prefix = str(p.get("prefix", "")).strip().lower()
            base_url = str(p.get("base_url", "")).strip()
            if not prefix or not base_url or prefix in seen:
                continue
            seen.add(prefix)
            # Round-10 R10-03: api_key persisted with Fernet (when
            # SHADOW_LOOM_SECRET_KEY is set). _decrypt_api_key returns
            # the plaintext for ENC1-prefixed blobs and passes
            # legacy plaintext rows through unchanged.
            cleaned_providers.append({
                "prefix": prefix,
                "base_url": base_url,
                "api_key": _decrypt_api_key(str(p.get("api_key", ""))),
                "is_local": bool(p.get("is_local", False)),
            })
        return {
            "default_model": (row.default_model or "").strip(),
            "stage_models": cleaned_stages,
            "custom_providers": cleaned_providers,
        }


def set_user_model_settings(
    user_id: int,
    *,
    default_model: str = "",
    stage_models: Optional[dict[str, str]] = None,
    custom_providers: Optional[list[dict]] = None,
) -> "UserModelSettingsRow":
    """Upsert per-user model preferences. Returns the persisted row.

    All arguments are optional — pass the subset you want to update.
    Empty / ``None`` values clear the override. Unknown stage keys are
    silently dropped. Custom providers are de-duplicated by ``prefix``
    (last occurrence wins).
    """
    import json as _json

    stage_models = stage_models or {}
    custom_providers = custom_providers or []

    # Sanitise stage models.
    cleaned_stages: dict[str, str] = {}
    for k, v in stage_models.items():
        if k in _RECOGNISED_STAGE_KEYS and isinstance(v, str) and v.strip():
            cleaned_stages[k] = v.strip()

    # Sanitise providers.
    cleaned_providers: list[dict] = []
    seen: set[str] = set()
    for p in custom_providers:
        if not isinstance(p, dict):
            raise ValueError("custom_providers entries must be dicts")
        prefix = str(p.get("prefix", "")).strip().lower()
        base_url = str(p.get("base_url", "")).strip()
        if not prefix or not base_url:
            continue
        if prefix in seen:
            # Last wins — drop earlier dup.
            cleaned_providers = [x for x in cleaned_providers if x["prefix"] != prefix]
        seen.add(prefix)
        # Round-10 R10-03: encrypt the secret before it ever lands in
        # the JSON blob so neither DB dumps nor backups leak it.
        cleaned_providers.append({
            "prefix": prefix,
            "base_url": base_url,
            "api_key": _encrypt_api_key(str(p.get("api_key", ""))),
            "is_local": bool(p.get("is_local", False)),
        })

    with get_session() as s:
        row = s.get(UserModelSettingsRow, user_id)
        if row is None:
            row = UserModelSettingsRow(
                user_id=user_id,
                default_model=default_model.strip(),
                stage_models_json=_json.dumps(cleaned_stages),
                custom_providers_json=_json.dumps(cleaned_providers),
            )
            s.add(row)
        else:
            row.default_model = default_model.strip()
            row.stage_models_json = _json.dumps(cleaned_stages)
            row.custom_providers_json = _json.dumps(cleaned_providers)
            row.updated_at = datetime.now(timezone.utc)
        s.commit()
        s.refresh(row)
        return row

