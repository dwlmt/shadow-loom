# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Seed pre-built example world models into the database at startup.

Each module in :mod:`example_worlds` exposes a top-level
``world_state: WorldStateV1``.  On UI startup we ensure one project
per example exists under the built-in *example* user, with the
world state stored as version 0.

When a user "selects" an example from the dashboard we fork that
project into the user's own account (re-using the existing
:func:`shadow_loom.db.fork_project`) instead of re-running ingestion.

The seeder is *idempotent* and has a cheap fast-path: it counts the
``.py`` fixture files on disk and compares against the number of
example projects already present in the DB.  If they match we skip
all module imports entirely.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from shadow_loom import db
from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)

_EXAMPLES_PACKAGE = "example_worlds"


def _sample_plots_dir() -> Path | None:
    """Locate the directory containing ``<slug>.txt`` sample plots.

    Tries, in order:
      1. The installed ``sample_plots`` package (works for ``pip install .``
         in production containers — files travel as package data).
      2. The repo-root ``sample_plots/`` directory two levels up from
         this file (works for editable installs and direct ``python``
         execution from the repo).
      3. ``$PWD/sample_plots`` as a last-resort fallback.
    Returns ``None`` if none of these exist.
    """
    try:
        pkg = importlib.import_module("sample_plots")
        pkg_path = Path(next(iter(pkg.__path__)))
        if pkg_path.is_dir():
            return pkg_path
    except (ImportError, StopIteration, AttributeError):
        pass

    repo_local = Path(__file__).resolve().parents[1] / "sample_plots"
    if repo_local.is_dir():
        return repo_local

    cwd_local = Path.cwd() / "sample_plots"
    if cwd_local.is_dir():
        return cwd_local

    return None


def _load_sample_plot(slug: str) -> str | None:
    """Return the contents of sample_plots/<slug>.txt if it exists."""
    base = _sample_plots_dir()
    if base is None:
        return None
    path = base / f"{slug}.txt"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("[examples] Failed to read %s", path)
        return None


def _fixture_slugs() -> list[str]:
    """Return the slug (module stem) of every fixture file on disk.

    Reads the directory directly — no module imports — so this is cheap
    enough to call on every startup.
    """
    try:
        pkg = importlib.import_module(_EXAMPLES_PACKAGE)
    except ImportError:
        return []
    pkg_dir = Path(next(iter(pkg.__path__)))
    return sorted(
        p.stem for p in pkg_dir.glob("*.py")
        if not p.stem.startswith("_")
    )


def _title_from_slug(slug: str) -> str:
    return slug.replace("_", " ").title()


def _load_world_state(slug: str) -> WorldStateV1 | None:
    mod_name = f"{_EXAMPLES_PACKAGE}.{slug}"
    try:
        mod = importlib.import_module(mod_name)
    except Exception:  # noqa: BLE001 — bad fixture shouldn't kill startup
        logger.exception("[examples] Failed to import %s", mod_name)
        return None
    ws = getattr(mod, "world_state", None)
    if not isinstance(ws, WorldStateV1):
        return None
    return ws


def _reconcile_existing(example_user_id: int, expected_names: set[str]) -> None:
    """Fix up legacy seeded rows so the live UI invariants always hold.

    * Force every example project to ``is_public=False`` (older seeder
      versions wrote them as public, which makes the access-control
      invariants harder to reason about).
    * Delete duplicate rows for the same example name, keeping the
      lowest id (and re-pointing the children of any duplicates is
      unnecessary because seeded rows have no forks of their own).
    """
    from sqlmodel import select
    from shadow_loom.db import (
        ProjectRow, VersionRow, ActivityRow, ProjectStarRow,
        ProjectMemberRow, get_session,
    )

    with get_session() as s:
        rows = s.exec(
            select(ProjectRow).where(ProjectRow.owner_id == example_user_id)
        ).all()

        # Group by name (only names we still expect).
        by_name: dict[str, list] = {}
        for r in rows:
            if r.name in expected_names:
                by_name.setdefault(r.name, []).append(r)

        dirty = False
        for name, group in by_name.items():
            group.sort(key=lambda r: r.id)
            keeper, *dups = group
            if keeper.is_public:
                keeper.is_public = False
                dirty = True
            # Backfill raw_text from the matching sample_plots fixture so
            # legacy seeded rows (created before raw_text was wired in)
            # show source text in the Story tab.
            if not keeper.raw_text:
                slug = name.lower().replace(" ", "_")
                plot = _load_sample_plot(slug)
                if plot:
                    keeper.raw_text = plot
                    dirty = True
            for dup in dups:
                # Cascade-delete dependent rows for the duplicate.
                for ver in s.exec(
                    select(VersionRow).where(VersionRow.project_id == dup.id)
                ).all():
                    s.delete(ver)
                for act in s.exec(
                    select(ActivityRow).where(ActivityRow.project_id == dup.id)
                ).all():
                    s.delete(act)
                for star in s.exec(
                    select(ProjectStarRow).where(ProjectStarRow.project_id == dup.id)
                ).all():
                    s.delete(star)
                for mem in s.exec(
                    select(ProjectMemberRow).where(ProjectMemberRow.project_id == dup.id)
                ).all():
                    s.delete(mem)
                s.delete(dup)
                dirty = True
                logger.info(
                    "[examples] Removed duplicate example project '%s' (id=%s)",
                    name, dup.id,
                )

        # Also drop any example projects whose name is no longer expected
        # (e.g. fixture renamed/removed).
        for r in rows:
            if r.name not in expected_names:
                for ver in s.exec(
                    select(VersionRow).where(VersionRow.project_id == r.id)
                ).all():
                    s.delete(ver)
                s.delete(r)
                dirty = True
                logger.info(
                    "[examples] Removed obsolete example project '%s' (id=%s)",
                    r.name, r.id,
                )

        if dirty:
            s.commit()


def seed_examples() -> int:
    """Ensure one example project per fixture exists for the example user.

    Idempotent.  Skips entirely when every fixture already has a
    matching project (the common steady-state path on every restart).
    Returns the number of projects newly created.
    """
    slugs = _fixture_slugs()
    if not slugs:
        return 0

    example_user = db.ensure_example_user()
    expected_names = {_title_from_slug(s) for s in slugs}

    # Heal any legacy/duplicate rows from older seeder versions before
    # checking the fast-path; otherwise stale public duplicates can
    # leak into the user-visible project listing.
    _reconcile_existing(example_user.id, expected_names)

    existing_names = {
        p["name"] for p in db.list_example_projects()
    }

    missing = expected_names - existing_names
    if not missing:
        logger.debug(
            "[examples] All %d examples already seeded — skipping.",
            len(expected_names),
        )
        return 0

    created = 0
    for slug in slugs:
        name = _title_from_slug(slug)
        if name not in missing:
            continue
        ws = _load_world_state(slug)
        if ws is None:
            continue
        try:
            proj = db.create_project(
                name=name,
                owner_id=example_user.id,
                description=f"Example world model: {name}",
                is_public=False,
                raw_text=_load_sample_plot(slug),
            )
            db.save_version(
                project_id=proj.id,
                world_state_json=ws.model_dump_json(),
                source="example",
                description="Pre-built example world model",
                user_id=example_user.id,
                version=0,
            )
            created += 1
            logger.info("[examples] Seeded %s (project_id=%s)", name, proj.id)
        except Exception:  # noqa: BLE001
            logger.exception("[examples] Failed to seed %s", name)

    if created:
        logger.info("[examples] Seeded %d new example projects", created)
    return created
