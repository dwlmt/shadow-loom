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
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    version: int = Field(default=0)
    ancestor_id: Optional[int] = Field(default=None, foreign_key="versions.id")

    source: str = Field(default="ingestion", max_length=64)
    description: Optional[str] = Field(default=None, max_length=512)
    label: Optional[str] = Field(default=None, max_length=128)
    is_bookmarked: bool = Field(default=False)

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
    """Activity feed entries for a project."""

    __tablename__ = "activities"

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
) -> VersionRow:
    """Persist a new version node in the version tree.

    If *version* is None it is auto-assigned as the next monotonic
    integer for the project.  Auto-assignment is retried up to a few
    times on ``IntegrityError`` to absorb concurrent writers racing on
    the (project_id, version) unique constraint.
    """
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
                    "label": r.label,
                    "is_bookmarked": r.is_bookmarked,
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


def get_all_prose(project_id: int) -> list[dict]:
    """Return all versions with prose, ordered by version number.

    Useful for exporting the full story.
    """
    with get_session() as s:
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
