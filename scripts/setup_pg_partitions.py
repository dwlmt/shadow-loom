# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Operator script to set up / extend Postgres partitions for log tables.

Run modes
---------

``--ensure``
    (Default.) Create the default partition + monthly partitions
    around ``now`` for every table in ``_PARTITIONED_TABLES``. Safe to
    run on every deploy and on a cron schedule (e.g. monthly) to roll
    new partitions forward. This is the same code path that
    :func:`shadow_loom.db.ensure_pg_partitions` runs at startup.

``--migrate``
    Convert legacy non-partitioned tables to partitioned tables. This:

    1. Renames ``<table>`` -> ``<table>_legacy``.
    2. Re-creates ``<table>`` as a partitioned table by re-running
       ``SQLModel.metadata.create_all`` for that table only.
    3. Creates the surrounding monthly partitions.
    4. Copies rows back: ``INSERT INTO <table> SELECT * FROM <table>_legacy``.
    5. Leaves the ``_legacy`` table in place for manual verification /
       drop.

    Requires a maintenance window — the table is unavailable for the
    duration of the copy. Skip this if you don't have a Postgres
    backend or the tables don't exist yet (fresh deploys are already
    partitioned by ``init_db``).

Usage
-----

::

    python scripts/setup_pg_partitions.py --ensure
    python scripts/setup_pg_partitions.py --migrate
    python scripts/setup_pg_partitions.py \
        --ensure --months-back 6 --months-forward 12

The database URL is read from ``$SHADOW_LOOM_DATABASE_URL`` (or
``$DATABASE_URL`` as a fallback) — same env var the application uses.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sqlalchemy import inspect, text
from sqlmodel import SQLModel

from shadow_loom.db import (
    _PARTITIONED_TABLES,
    _convert_table_to_partitioned,
    ensure_pg_partitions,
    init_db,
    get_engine,
)

logger = logging.getLogger("setup_pg_partitions")


def _resolve_url() -> str:
    url = os.environ.get("SHADOW_LOOM_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        sys.exit(
            "ERROR: set SHADOW_LOOM_DATABASE_URL (or DATABASE_URL) to a "
            "postgres:// connection string before running this script."
        )
    if not (url.startswith("postgres://") or url.startswith("postgresql")):
        sys.exit(f"ERROR: this script only operates on Postgres URLs (got {url!r}).")
    return url


def _is_partitioned(engine, table: str) -> bool:  # noqa: ANN001
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT relkind FROM pg_class WHERE relname = :name AND "
                "relnamespace = (SELECT oid FROM pg_namespace "
                "WHERE nspname = current_schema())"
            ),
            {"name": table},
        ).first()
    return row is not None and row[0] == "p"


def _migrate_table(engine, table: str) -> None:  # noqa: ANN001
    """Rename legacy table, re-create partitioned, copy data back."""
    insp = inspect(engine)
    if table not in insp.get_table_names():
        logger.info("[migrate] %s does not exist — nothing to migrate.", table)
        return
    if _is_partitioned(engine, table):
        logger.info("[migrate] %s already partitioned — skipping.", table)
        return

    legacy = f"{table}_legacy"
    logger.warning(
        "[migrate] Converting %s -> partitioned (legacy copy preserved as %s).",
        table, legacy,
    )

    sqla_table = SQLModel.metadata.tables.get(table)
    if sqla_table is None:
        sys.exit(f"ERROR: {table!r} is not registered in SQLModel.metadata.")

    with engine.begin() as conn:
        # 1. Snapshot legacy schema before touching it.
        legacy_cols_before = {
            c["name"] for c in inspect(engine).get_columns(table)
        }

        # 2. Rename the existing table out of the way.
        conn.execute(text(f'ALTER TABLE "{table}" RENAME TO "{legacy}"'))

        # 3. Create the partitioned replacement using the shared DDL
        #    builder so column shape, FKs and indexes match what the
        #    application will use at runtime.
        _convert_table_to_partitioned(conn, table)

    # 4. Create monthly partitions before copying data so every row
    #    finds a home (the DEFAULT partition catches outliers).
    ensure_pg_partitions(engine)

    # 5. Copy data — only columns present on both sides.
    with engine.begin() as conn:
        new_cols = {c.name for c in sqla_table.columns}
        common = [c for c in new_cols if c in legacy_cols_before]
        col_list = ", ".join(f'"{c}"' for c in common)
        result = conn.execute(
            text(
                f'INSERT INTO "{table}" ({col_list}) '
                f'SELECT {col_list} FROM "{legacy}"'
            )
        )
        # Resync the IDENTITY sequence so future inserts don't collide
        # with copied ids. ``pg_get_serial_sequence`` returns NULL for
        # IDENTITY columns; use the standard ``ALTER TABLE … ALTER
        # COLUMN id RESTART WITH …`` form instead.
        max_id = conn.execute(
            text(f'SELECT COALESCE(MAX("id"), 0) + 1 FROM "{table}"')
        ).scalar_one()
        try:
            conn.execute(
                text(
                    f'ALTER TABLE "{table}" ALTER COLUMN "id" '
                    f'RESTART WITH {int(max_id)}'
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[migrate] Could not reset IDENTITY for %s.id; new "
                "inserts may collide with copied rows.",
                table,
            )
        logger.info(
            "[migrate] Copied %s row(s) from %s to %s.",
            result.rowcount, legacy, table,
        )
    logger.warning(
        "[migrate] %s migrated. Verify, then DROP TABLE %s manually.",
        table, legacy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Create default + monthly partitions (default if no flag given).",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Convert legacy non-partitioned tables to partitioned tables. "
        "Requires a maintenance window.",
    )
    parser.add_argument("--months-back", type=int, default=3)
    parser.add_argument("--months-forward", type=int, default=3)
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(_PARTITIONED_TABLES),
        help="Subset of partitioned tables to operate on.",
    )
    args = parser.parse_args()

    if not (args.ensure or args.migrate):
        args.ensure = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    url = _resolve_url()
    init_db(url)
    engine = get_engine()

    if args.migrate:
        for table in args.tables:
            _migrate_table(engine, table)

    if args.ensure:
        ensure_pg_partitions(
            engine,
            months_back=args.months_back,
            months_forward=args.months_forward,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
