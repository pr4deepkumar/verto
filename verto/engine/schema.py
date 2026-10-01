"""
engine/schema.py
~~~~~~~~~~~~~~~~
Schema loading and schema-linking logic.

Schema-linking scores each table against the user's question using token
overlap, so the model only sees the most relevant tables rather than an
entire (potentially huge) database schema. Table-name matches are weighted
2× relative to column-name matches because they are stronger signals.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_tables_raw(tables_json_path: str | Path) -> list[dict]:
    """Return the raw list of database descriptors from Spider's tables.json."""
    with open(tables_json_path, "r") as f:
        return json.load(f)


def _format_table(db_info: dict, table_idx: int) -> str:
    """One table as a schema line, e.g.
        Table: singer (Singer_ID (number) [PK], Name (text), Age (number))

    Primary keys are annotated with [PK] and foreign keys with [FK -> table.col].
    """
    column_names = db_info["column_names_original"]
    table_names = db_info["table_names_original"]
    columns: list[str] = []
    for col_idx, (tab_idx, col_name) in enumerate(column_names):
        if tab_idx != table_idx:
            continue

        col_type = db_info["column_types"][col_idx]
        is_pk = col_idx in db_info.get("primary_keys", [])
        pk_marker = " [PK]" if is_pk else ""

        fk_marker = ""
        for fk_col1, fk_col2 in db_info.get("foreign_keys", []):
            if col_idx == fk_col1:
                other_tab = table_names[column_names[fk_col2][0]]
                other_col = column_names[fk_col2][1]
                fk_marker = f" [FK -> {other_tab}.{other_col}]"
                break

        columns.append(f"{col_name} ({col_type}){pk_marker}{fk_marker}")

    return f"Table: {table_names[table_idx]} ({', '.join(columns)})"


def format_schemas(tables_data: list[dict]) -> dict[str, str]:
    """Map each db_id in Spider's tables.json data to its schema text, one line per table."""
    return {
        db_info["db_id"]: "\n".join(
            _format_table(db_info, table_idx) for table_idx in range(len(db_info["table_names_original"]))
        )
        for db_info in tables_data
    }


def load_and_format_schemas(tables_json_path: str | Path) -> dict[str, str]:
    """Parse Spider's tables.json and return a dict mapping db_id → schema text."""
    return format_schemas(load_tables_raw(tables_json_path))


# ---------------------------------------------------------------------------
# Schema-linking
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokenisation with basic stemming for overlap scoring."""
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = set(raw_tokens)
    for t in raw_tokens:
        stemmed = _stem(t)
        if len(stemmed) >= 2:
            tokens.add(stemmed)
    return tokens


def schema_link(
    question: str,
    db_id: str,
    tables_data_raw: list[dict],
    full_schema_dict: dict[str, str],
    top_k_tables: int = 3,
) -> str:
    """Return a schema string containing only the most relevant tables.

    Tables are scored by token overlap with the question (table name matches
    count 2×, column name matches count 1×). If no tokens overlap at all, the
    full schema is returned unchanged to avoid giving the model no context.

    Args:
        question: The natural-language question.
        db_id: Spider database identifier.
        tables_data_raw: The raw list loaded from tables.json.
        full_schema_dict: Pre-formatted full schemas keyed by db_id.
        top_k_tables: Maximum number of tables to include.

    Returns:
        A multi-line schema string with only the selected tables.
    """
    q_tokens = _tokenize(question)
    db_info = next((d for d in tables_data_raw if d["db_id"] == db_id), None)

    if db_info is None:
        return full_schema_dict.get(db_id, "")

    column_names = db_info["column_names_original"]
    table_names = db_info["table_names_original"]

    table_scores: dict[int, float] = defaultdict(float)
    for table_idx, table_name in enumerate(table_names):
        table_scores[table_idx] += len(q_tokens & _tokenize(table_name)) * 2
        for col_idx, (tab_idx, col_name) in enumerate(column_names):
            if tab_idx == table_idx:
                table_scores[table_idx] += len(q_tokens & _tokenize(col_name))

    # Fall back to full schema if nothing matches
    if not table_scores or max(table_scores.values()) == 0:
        return full_schema_dict.get(db_id, "")

    top_tables = sorted(table_scores, key=table_scores.get, reverse=True)[:top_k_tables]  # type: ignore[arg-type]
    return "\n".join(_format_table(db_info, table_idx) for table_idx in sorted(top_tables))
