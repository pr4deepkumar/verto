"""
verto/api/models.py
~~~~~~~~~~~~~~~~~~~
Pydantic request / response schemas for the VERTO FastAPI backend.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for POST /api/query."""

    question: str = Field(..., min_length=1, max_length=1000, description="Plain-English question.")
    db_id: str = Field(..., min_length=1, max_length=128, description="Spider database identifier.")
    use_schema_linking: bool = Field(
        True, description="Pass only the most relevant tables to the model."
    )
    use_error_recovery: bool = Field(
        True, description="If the top query fails, run the model's other candidates in order."
    )


class QueryResponse(BaseModel):
    """Response from POST /api/query."""

    sql: str | None = Field(None, description="Generated SQL query.")
    columns: list[str] | None = Field(
        None, description="Column names of the result, or null on error."
    )
    result: list[list[Any]] | None = Field(
        None, description="Query result rows (each row is a list of values)."
    )
    attempts: int = Field(1, description="Number of generation attempts used.")
    error: str | None = Field(None, description="Error message, or null on success.")


class DatabasesResponse(BaseModel):
    """Response from GET /api/databases."""

    databases: list[str] = Field(..., description="Sorted list of available db_ids.")
    count: int


class HealthResponse(BaseModel):
    """Response from GET /api/health."""

    status: str
    device: str
    checkpoint_loaded: bool
    database_count: int
