"""
engine/inference.py
~~~~~~~~~~~~~~~~~~~
NLDIEngine — the single object used by the CLI, the API server and the evaluation in
scripts/train.py.

It owns the model, tokeniser, schema data, and database directory, and
exposes a simple ``query()`` method that handles schema-linking, SQL
generation, error recovery, and execution in one call.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from .executor import execute_sql
from .schema import format_schemas, load_tables_raw, schema_link

logger = logging.getLogger(__name__)


def build_prompt(question: str, schema: str) -> str:
    """The input the model is trained on (scripts/train.py) and queried with."""
    return f"translate to SQL: {question} \n{schema}"


def run_with_recovery(
    generate: Callable[[str], list[str]],
    question: str,
    schema: str,
    db_path: str | Path,
    use_error_recovery: bool = True,
) -> dict[str, Any]:
    """Generate candidate queries with *generate* (best first) and keep the first that runs.

    The top candidate runs first. If SQLite rejects it and error recovery is on, the other
    candidates run in order until one works; with recovery off, the top one is the only try.
    Either way the model is called once. Returns the dict described in :meth:`NLDIEngine.query`.
    """
    candidates = list(dict.fromkeys(generate(build_prompt(question, schema))))  # drop duplicates
    tries = candidates if use_error_recovery else candidates[:1]
    first_error = None
    for attempt, sql in enumerate(tries, 1):
        result = execute_sql(sql, db_path)
        if not isinstance(result, str):
            columns, rows = result
            return {"sql": sql, "columns": columns, "result": [list(row) for row in rows],
                    "attempts": attempt, "error": None}
        first_error = first_error or result

    # Nothing ran: report the model's top query and why it failed.
    return {"sql": candidates[0], "columns": None, "result": None, "attempts": len(tries), "error": first_error}


class NLDIEngine:
    """Orchestrates the full text-to-SQL pipeline.

    Args:
        checkpoint_path: Path to ``best_model.pt``.  May be ``None`` or
            point to a non-existent file — in that case the base model is used
            without a fine-tuned adapter, which is useful for testing the scaffold.
        spider_dir: Root of the Spider dataset, containing ``tables.json``
            and a ``database/`` subdirectory with one SQLite file per DB.
        device: Torch device string, e.g. ``"cuda"``, ``"mps"`` or ``"cpu"``.
            Defaults to auto-detection.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        spider_dir: str | Path = "./data/spider",
        device: str | None = None,
    ) -> None:
        self.spider_dir = Path(spider_dir)
        self.db_dir = self.spider_dir / "database"
        tables_json = self.spider_dir / "tables.json"
        if not tables_json.exists():
            raise FileNotFoundError(
                f"tables.json not found at {tables_json}. "
                "Point VERTO_SPIDER_DIR at your Spider dataset root."
            )

        logger.info("Loading schemas from %s …", tables_json)
        self._tables_raw = load_tables_raw(tables_json)
        self._schema_dict = format_schemas(self._tables_raw)

        from .model import load_model  # imported here: torch is only needed once a model loads

        self.model, self.tokenizer = load_model(checkpoint_path, device)
        self.device = next(self.model.parameters()).device
        self._lock = threading.Lock()  # one generation at a time on the shared model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_databases(self) -> list[str]:
        """Return all database IDs available in the Spider dataset."""
        return sorted(self._schema_dict.keys())

    def schema_for(self, question: str, db_id: str, use_schema_linking: bool = True) -> str:
        """The schema text the prompt for *question* gets: linked tables or the full schema."""
        if not use_schema_linking:
            return self._schema_dict[db_id]
        return schema_link(question, db_id, self._tables_raw, self._schema_dict)

    def query(
        self,
        question: str,
        db_id: str,
        use_schema_linking: bool = True,
        use_error_recovery: bool = True,
    ) -> dict[str, Any]:
        """Translate *question* into SQL and execute it against *db_id*.

        Args:
            question: Plain-English question.
            db_id: Spider database identifier (see :meth:`list_databases`).
            use_schema_linking: If ``True``, only the most relevant tables
                are included in the prompt.
            use_error_recovery: If ``True`` and the model's top query fails,
                its other beam-search candidates run in order until one works.
                If ``False``, the top query is the only one that runs.

        Returns:
            A dict with keys:
                ``sql``      — the SQL that ran, or the top candidate if none did.
                ``columns``  — the result's column names on success, else ``None``.
                ``result``   — list of result rows (each row is a list) on
                               success, or ``None`` on error.
                ``attempts`` — number of candidate queries run (1 = the top one worked).
                ``error``    — error message string, or ``None`` on success.
        """
        if db_id not in self._schema_dict:
            return {
                "sql": None,
                "columns": None,
                "result": None,
                "attempts": 0,
                "error": f'Unknown database "{db_id}". Use list_databases() to see available IDs.',
            }

        from .model import generate_candidates

        schema = self.schema_for(question, db_id, use_schema_linking)
        db_path = self.db_dir / db_id / f"{db_id}.sqlite"

        def generate(prompt: str) -> list[str]:
            return generate_candidates(self.model, self.tokenizer, prompt, self.device)

        with self._lock:
            return run_with_recovery(generate, question, schema, db_path, use_error_recovery)
