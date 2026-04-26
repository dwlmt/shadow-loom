"""Shared SQLModel persistence layer for Shadow-Loom.

Used by both the NiceGUI web UI and the MCP server.
Standalone — no UI imports.  Configure via ``init_db(database_url)``.
"""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

# =====================================================================
# ORM Base
# =====================================================================

EXAMPLE_USER_PROVIDER_ID = "local:example"

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
    email: Optional[str] = Field(default=None, max_length=256)
    avatar_url: Optional[str] = Field(default=None, max_length=512)
    is_example: bool = Field(default=False)
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
    raw_text: Optional[str] = Field(default=None, sa_column=Column(Text))
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
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    version: int = Field(default=0)
    ancestor_id: Optional[int] = Field(default=None, foreign_key="versions.id")

    source: str = Field(default="ingestion", max_length=64)
    description: Optional[str] = Field(default=None, max_length=512)

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
    _engine = create_engine(database_url, echo=False)
    SQLModel.metadata.create_all(_engine)
    ensure_example_user()
    logger.info("[DB] Tables initialised on %s", database_url)


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


# =====================================================================
# User CRUD
# =====================================================================


def upsert_user(
    provider: str,
    provider_id: str,
    username: str,
    email: str | None = None,
    avatar_url: str | None = None,
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
            )
            s.add(row)
        else:
            row.username = username
            row.email = email
            row.avatar_url = avatar_url
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
) -> ProjectRow:
    with get_session() as s:
        row = ProjectRow(
            name=name,
            owner_id=owner_id,
            label=label,
            raw_text=raw_text,
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

    Includes the user's own projects **plus** all projects owned by the
    example user (``is_example=True``).
    """
    example_id = get_example_user_id()

    with get_session() as s:
        stmt = select(ProjectRow)

        if user_id is not None:
            owner_ids = [user_id]
            if example_id is not None:
                owner_ids.append(example_id)
            stmt = stmt.where(
                or_(
                    ProjectRow.owner_id.in_(owner_ids),
                    ProjectRow.owner_id.is_(None),
                )
            )
        # else: return all projects (admin / unauthenticated mode)

        rows = s.exec(stmt.order_by(ProjectRow.updated_at.desc())).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "label": r.label,
                "owner_id": r.owner_id,
                "is_example": (r.owner_id == example_id) if example_id else False,
                "updated_at": str(r.updated_at),
                "version_count": len(r.versions),
            }
            for r in rows
        ]


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
) -> VersionRow:
    """Persist a new version node in the version tree.

    If *version* is None it is auto-assigned as the next monotonic
    integer for the project.
    """
    with get_session() as s:
        if version is None:
            version = _next_version_number(s, project_id)

        row = VersionRow(
            project_id=project_id,
            version=version,
            ancestor_id=ancestor_id,
            source=source,
            description=description,
            world_state_json=world_state_json,
            changeset_json=changeset_json,
            raw_query=raw_query,
            parsed_query_json=parsed_query_json,
            prose=prose,
            user_id=user_id,
        )
        s.add(row)

        # Touch project updated_at
        proj = s.get(ProjectRow, project_id)
        if proj:
            proj.updated_at = datetime.now(timezone.utc)

        s.commit()
        s.refresh(row)
        return row


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
                "has_prose": r.prose is not None,
                "has_changeset": r.changeset_json is not None,
                "user_id": r.user_id,
                "created_at": str(r.created_at),
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
                    changeset_summary = {
                        "events_added": cs.get("events_added", 0),
                        "causal_edges_added": cs.get("causal_edges_added", 0),
                        "entity_updates_applied": cs.get("entity_updates_applied", 0),
                    }
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(
                {
                    "id": r.id,
                    "version": r.version,
                    "ancestor_id": r.ancestor_id,
                    "source": r.source,
                    "description": r.description,
                    "changeset_summary": changeset_summary,
                    "user_id": r.user_id,
                    "created_at": str(r.created_at),
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
