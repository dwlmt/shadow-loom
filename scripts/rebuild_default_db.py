#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Wipe ``shadow_loom.db`` and re-seed it with the bundled example worlds.

For every module in ``example_worlds/`` this script:

1. Imports its ``world_state: WorldStateV1`` fixture.
2. Reads the matching ``sample_plots/<name>.txt`` (when present) and
   stores it as ``ProjectRow.raw_text`` so the Story tab has source.
3. Creates a ``ProjectRow`` owned by the built-in *example* user (so it
   surfaces via :func:`list_example_projects` and is copied into a
   real user's account on selection — never shown directly in their
   project list) and writes the fixture as the project's first
   ``VersionRow`` with ``source="seed"``.

Run with::

    python scripts/rebuild_default_db.py            # uses settings.core.database_url
    python scripts/rebuild_default_db.py --db-url sqlite:///custom.db
    python scripts/rebuild_default_db.py --keep     # do not delete an existing sqlite file
"""
from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import sys
from pathlib import Path

# Make the project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import example_worlds  # noqa: E402
from shadow_loom.db import (  # noqa: E402
    create_project,
    ensure_example_user,
    init_db,
    save_version,
)
from shadow_loom.models import WorldStateV1  # noqa: E402
from shadow_loom.settings import get_settings  # noqa: E402

logger = logging.getLogger("rebuild_default_db")

SAMPLE_PLOTS_DIR = PROJECT_ROOT / "sample_plots"


def _humanize(module_name: str) -> str:
    """``a_fish_called_wanda`` -> ``A Fish Called Wanda``."""
    small = {"a", "an", "the", "of", "and", "in", "on", "for"}
    parts = module_name.split("_")
    out: list[str] = []
    for i, p in enumerate(parts):
        if i > 0 and p in small:
            out.append(p)
        else:
            out.append(p.capitalize())
    return " ".join(out)


def _discover_world_modules() -> list[str]:
    names: list[str] = []
    for info in pkgutil.iter_modules(example_worlds.__path__):
        if info.ispkg:
            continue
        if info.name.startswith("_"):
            continue
        names.append(info.name)
    names.sort()
    return names


def _maybe_delete_sqlite(database_url: str, *, keep: bool) -> None:
    if keep:
        return
    if not database_url.startswith("sqlite:///"):
        logger.info("Non-sqlite URL %r — skipping file delete.", database_url)
        return
    rel = database_url[len("sqlite:///") :]
    db_path = Path(rel)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if db_path.exists():
        logger.info("Removing existing database file: %s", db_path)
        db_path.unlink()
    else:
        logger.info("No existing database file at %s — nothing to delete.", db_path)


def _load_raw_text(module_name: str) -> str | None:
    plot_file = SAMPLE_PLOTS_DIR / f"{module_name}.txt"
    if plot_file.is_file():
        return plot_file.read_text(encoding="utf-8")
    return None


def _seed_one(module_name: str, example_user_id: int) -> tuple[int, int]:
    """Return ``(project_id, version_row_id)`` for the seeded world."""
    module = importlib.import_module(f"example_worlds.{module_name}")
    ws = getattr(module, "world_state", None)
    if not isinstance(ws, WorldStateV1):
        raise TypeError(
            f"example_worlds.{module_name} does not export a WorldStateV1 'world_state'"
        )

    raw_text = _load_raw_text(module_name)
    pretty = _humanize(module_name)

    proj = create_project(
        name=pretty,
        owner_id=example_user_id,
        label=module_name,
        raw_text=raw_text,
        description=f"Bundled example world: {pretty}",
        is_public=False,
    )

    version = save_version(
        project_id=proj.id,
        world_state_json=ws.model_dump_json(),
        source="seed",
        description=f"Initial seed of {pretty} from example_worlds.{module_name}",
        label="seed",
    )
    return proj.id, version.id


def rebuild(database_url: str | None = None, *, keep: bool = False) -> None:
    settings = get_settings()
    db_url = database_url or settings.core.database_url

    print(f"🛠  Rebuilding default database at: {db_url}")
    _maybe_delete_sqlite(db_url, keep=keep)

    init_db(db_url)
    print("✅ Schema initialised")

    example_user = ensure_example_user()
    print(f"👤 Example user id={example_user.id} ({example_user.username})")

    modules = _discover_world_modules()
    print(f"🌱 Seeding {len(modules)} example worlds…")

    failures: list[tuple[str, Exception]] = []
    for name in modules:
        try:
            pid, vid = _seed_one(name, example_user.id)
            print(f"   ✓ {name:<40s} project_id={pid} version_row_id={vid}")
        except Exception as exc:  # noqa: BLE001 — surface each failure, keep going
            failures.append((name, exc))
            print(f"   ✗ {name:<40s} FAILED: {exc}")

    print()
    if failures:
        print(f"⚠️  {len(failures)} world(s) failed to seed:")
        for name, exc in failures:
            print(f"     - {name}: {exc!r}")
        sys.exit(1)
    print(f"🎉 Done. Seeded {len(modules)} worlds into {db_url}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=None,
        help="Override database URL (default: settings.core.database_url).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete an existing sqlite file before re-seeding.",
    )
    args = parser.parse_args()
    rebuild(database_url=args.db_url, keep=args.keep)


if __name__ == "__main__":
    main()
