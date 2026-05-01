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
    ensure_example_user()
    logger.info("[DB] Tables initialised on %s", database_url)


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
) -> VersionRow:
    """Copy a shadow-branch version onto the factual mainline as a new version.

    Creates a *new* mainline VersionRow whose ``world_state_json`` and
    ``prose`` come verbatim from the shadow source, but whose
    ``ancestor_id`` points at the current factual head and whose
    ``world_id`` is forced to ``'factual'``. The shadow source is left
    untouched so the fork remains browsable.

    Raises ``VersionMutationError`` if the source version does not exist
    or already lives on the factual mainline (use the standard
    save/branch flows for those cases).
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

    promoted_desc = description or (
        f"Promoted shadow v{src.version}"
        + (f" ({src.branch_label})" if src.branch_label else "")
        + " to factual mainline"
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
        branch_label=None,
    )


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
