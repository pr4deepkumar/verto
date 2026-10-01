"""
engine/executor.py
~~~~~~~~~~~~~~~~~~
SQLite query execution and SQL normalisation utilities.

Generated SQL runs on a read-only connection that cannot attach other database files,
under a time limit and a row cap.
"""

import re
import sqlite3
import time
from pathlib import Path

TIMEOUT_SECONDS = 10.0
MAX_ROWS = 10_000


def _deny_attach(action: int, *_args) -> int:
    return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK


def execute_sql(
    sql: str,
    db_path: str | Path,
    timeout: float = TIMEOUT_SECONDS,
    max_rows: int = MAX_ROWS,
) -> tuple[list[str], list[tuple]] | str:
    """Run *sql* against the SQLite database at *db_path*.

    Returns:
        ``(columns, rows)`` on success: the result's column names and up to *max_rows*
        rows, in the order SQLite returned them.
        An error string (starting with "Error:") on failure. Writes fail, because the
        connection is read-only, and so do queries still running after *timeout* seconds.
    """
    db_p = Path(db_path)
    if not db_p.exists():
        return f"Error: Database file does not exist at {db_path}"

    conn = None
    try:
        conn = sqlite3.connect(f"{db_p.resolve().as_uri()}?mode=ro", uri=True)
        conn.set_authorizer(_deny_attach)
        deadline = time.monotonic() + timeout
        # SQLite calls this every 10,000 virtual-machine steps; a non-zero return stops the query.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        return columns, cursor.fetchmany(max_rows)
    except sqlite3.OperationalError as exc:
        if str(exc) == "interrupted":
            return f"Error: query still running after {timeout:g} s"
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        if conn is not None:
            conn.close()


def normalize_sql(sql: str) -> str:
    """Normalise SQL for exact-match comparison.

    Strips leading/trailing whitespace, collapses internal whitespace,
    and lowercases the string so that trivial formatting differences
    don't count as mismatches.
    """
    return re.sub(r"\s+", " ", sql.strip()).lower()
