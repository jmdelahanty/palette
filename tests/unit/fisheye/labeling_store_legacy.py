"""Build pre-v9 labeling stores in tests: the same tables without state CHECKs."""

from __future__ import annotations

import re
import sqlite3

STATE_TABLES = (
    "labeling_tasks",
    "labeling_session_checkpoints",
    "labeling_checkpoint_apply_receipts",
)


def strip_state_checks(conn: sqlite3.Connection) -> None:
    """Rebuild the state tables without their v9 CHECK clauses, rows and indexes kept."""

    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF;")
    try:
        for table in STATE_TABLES:
            sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?;", (table,)).fetchone()[0]
            indexes = [row[0] for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL;",
                (table,),
            )]
            plain = re.sub(r",\s*CHECK \([^)]*\)\)", "", sql)
            plain = plain.replace(f'CREATE TABLE "{table}"', f'CREATE TABLE "{table}__old"', 1)
            assert "CHECK" not in plain, plain
            conn.execute(plain)
            conn.execute(f'INSERT INTO "{table}__old" SELECT * FROM "{table}";')
            conn.execute(f'DROP TABLE "{table}";')
            conn.execute(f'ALTER TABLE "{table}__old" RENAME TO "{table}";')
            for index_sql in indexes:
                conn.execute(index_sql)
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON;")


def as_v8(path) -> None:
    """Rewrite a current store into the v8 shape."""

    conn = sqlite3.connect(path)
    try:
        strip_state_checks(conn)
        conn.execute("UPDATE labeling_schema_meta SET value = '8' WHERE key = 'schema_version';")
        conn.commit()
    finally:
        conn.close()
