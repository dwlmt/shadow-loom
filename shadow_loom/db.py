# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared SQLModel persistence layer for Shadow-Loom.

Used by both the NiceGUI web UI and the MCP server.
Standalone — no UI imports.  Configure via ``init_db(database_url)``.
"""

import hashlib
import json
import logging
import secrets
from datetime import datetime, timezone
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
    user_id: int = Field(foreign_key="users.id")
    name: str = Field(max_length=128)
    key_hash: str = Field(max_length=128)
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
SCHEMA_VERSION_CURRENT: int = 3

# Ordered ledger of applied migrations: (version, name).
# Version 1 is the historical baseline (everything before this ledger
# existed); version 2 added world_id + branch_label columns to
# ``versions`` (handled by ``_run_lightweight_migrations``); version 3
# introduced Postgres RANGE partitioning + composite indexes on the
# high-volume log tables (activities, agent_call_logs, api_call_logs).
_SCHEMA_MIGRATIONS: list[tuple[int, str]] = [
    (1, "baseline"),
    (2, "versions.world_id+branch_label"),
    (3, "log-tables.partitioning+indexes"),
]


# =====================================================================
# Engine / Session factory
# =====================================================================

_engine = None


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
    if database_url.startswith("postgresql"):
        ensure_pg_partitions(_engine)
    _record_schema_versions(_engine)
    ensure_example_user()
    logger.info("[DB] Tables initialised on %s", database_url)


def _record_schema_versions(engine) -> None:  # noqa: ANN001
    """Stamp ``schema_versions`` so we can audit migration history.

    Inserts every entry from :data:`_SCHEMA_MIGRATIONS` that isn't
    already present. Idempotent: re-running just no-ops, so it's safe
    to call on every startup.
    """
    with Session(engine) as s:
        existing = set(s.exec(select(SchemaVersionRow.version)).all())
        added = 0
        for version, name in _SCHEMA_MIGRATIONS:
            if version in existing:
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

    if not statements:
        return

    with engine.begin() as conn:
        for sql in statements:
            try:
                conn.execute(text(sql))
            except Exception:  # noqa: BLE001
                # Concurrent migrators (multiple workers booting in
                # parallel) may race the ALTER. Re-inspect rather than
                # propagate so the second arrival just no-ops.
                logger.exception(
                    "[DB·migrate] Failed to apply: %s (treating as already applied).",
                    sql,
                )
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
) -> None:
    """Ensure log tables are partitioned and have monthly child tables.

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

    with engine.begin() as conn:
        for table in _PARTITIONED_TABLES:
            relkind = _table_relkind(conn, table)
            if relkind is None:
                # Table not created yet — nothing to do.
                continue

            if relkind != "p":
                # Try to auto-convert if empty (fresh-deploy path).
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
                        continue
                else:
                    logger.warning(
                        "[DB·partitions] Table %r has %d row(s) and is "
                        "NOT partitioned. Run "
                        "scripts/setup_pg_partitions.py --migrate to "
                        "convert (requires a maintenance window).",
                        table, count_row,
                    )
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

    logger.info(
        "[DB·partitions] Ensured monthly partitions for %d table(s) "
        "(window: -%d / +%d months).",
        len(_PARTITIONED_TABLES), months_back, months_forward,
    )


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


def update_project(
    project_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    label: str | None = None,
    is_public: bool | None = None,
    raw_text: str | None = None,
) -> Optional[ProjectRow]:
    with get_session() as s:
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
) -> Optional[ProjectRow]:
    """Fork a project: copy latest version to a new project owned by new_owner_id."""
    with get_session() as s:
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

        forked = ProjectRow(
            name=new_name or f"{source.name} (fork)",
            owner_id=new_owner_id,
            label=source.label,
            description=source.description,
            raw_text=source.raw_text,
            forked_from_id=source_project_id,
        )
        s.add(forked)
        s.flush()

        ver = VersionRow(
            project_id=forked.id,
            version=0,
            source="fork",
            description=f"Forked from project {source_project_id}",
            world_state_json=latest.world_state_json,
        )
        s.add(ver)
        s.commit()
        s.refresh(forked)
        return forked


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
) -> ProjectMemberRow:
    with get_session() as s:
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
        s.commit()
        s.refresh(row)
        return row


def remove_project_member(project_id: int, user_id: int) -> bool:
    with get_session() as s:
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
    with get_session() as s:
        existing = s.exec(
            select(ProjectStarRow).where(
                ProjectStarRow.project_id == project_id,
                ProjectStarRow.user_id == user_id,
            )
        ).first()
        proj = s.get(ProjectRow, project_id)
        if existing:
            s.delete(existing)
            if proj:
                proj.star_count = max(0, proj.star_count - 1)
            s.commit()
            return False
        else:
            s.add(ProjectStarRow(project_id=project_id, user_id=user_id))
            if proj:
                proj.star_count = proj.star_count + 1
            s.commit()
            return True


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


def _hash_api_key(raw_key: str) -> str:
    """SHA-256 hash for API key storage."""
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
    key_hash = _hash_api_key(raw_key)
    with get_session() as s:
        row = s.exec(
            select(ApiKeyRow).where(
                ApiKeyRow.key_hash == key_hash,
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
    # Cheap shape-check at the persistence boundary so corrupted
    # payloads are flagged early. Logged-only by default (warning) so
    # legacy / stub callers continue to work; controlled by the
    # ``SHADOW_LOOM_STRICT_PERSIST`` env var, which when set to
    # ``"1"``/``"true"`` upgrades the warning to a hard ``ValueError``.
    # Skipped entirely when ``accept_partial=True`` so explicit
    # "persist what we have" callers still get through.
    if not accept_partial:
        try:
            from shadow_loom.models import WorldStateV1
            WorldStateV1.model_validate_json(world_state_json)
        except Exception as exc:
            import os as _os
            strict = _os.environ.get("SHADOW_LOOM_STRICT_PERSIST", "").lower() in {"1", "true", "yes", "on"}
            msg = (
                f"save_version: world_state_json failed WorldStateV1 "
                f"validation at persistence boundary: {exc}"
            )
            if strict:
                raise ValueError(
                    msg + " (SHADOW_LOOM_STRICT_PERSIST=1; pass "
                    "accept_partial=True to bypass.)"
                ) from exc
            logger.warning(msg)
    # When an explicit version is supplied we honour it (single attempt).
    max_attempts = 1 if version is not None else 5
    last_err: Exception | None = None
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
    # A version is a *branch root* when it has no ancestor (mainline v0)
    # or when its ancestor lives on a different world_id.
    branches: list[dict] = []
    for r in rows:
        parent = by_id.get(r.ancestor_id) if r.ancestor_id is not None else None
        is_root = parent is None or parent.world_id != r.world_id
        if not is_root:
            continue
        # Walk forward along same-world_id direct descendants to find the head.
        head = r
        version_count = 1
        # Children are not pre-indexed; do a simple linear search per branch.
        # Branches are typically shallow so the cost stays small.
        while True:
            same_branch_children = [
                c for c in rows
                if c.ancestor_id == head.id and c.world_id == head.world_id
            ]
            if not same_branch_children:
                break
            # If a branch fans out (multiple children on the same world_id),
            # pick the highest-version child as the canonical head and stop —
            # downstream ``get_version_children`` exposes the rest.
            same_branch_children.sort(key=lambda c: c.version)
            head = same_branch_children[-1]
            version_count += 1
            if len(same_branch_children) > 1:
                break
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


def promote_branch(
    version_row_id: int,
    *,
    user_id: int | None = None,
    description: str | None = None,
    force: bool = False,
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
    return save_version(
        project_id=src.project_id,
        world_state_json=src.world_state_json,
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
    """BFS over the version tree to collect all descendants (inclusive)."""
    seen: set[int] = {root_id}
    frontier = [root_id]
    while frontier:
        children = s.exec(
            select(VersionRow).where(VersionRow.ancestor_id.in_(frontier))
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
            s.flush()
            # Delete deepest-first to satisfy the self-FK on ancestor_id.
            ordered = sorted(descendants, reverse=True)
            for vid in ordered:
                v = s.get(VersionRow, vid)
                if v is not None:
                    s.delete(v)
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

