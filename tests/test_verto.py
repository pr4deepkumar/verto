"""
tests/test_verto.py
~~~~~~~~~~~~~~~~~~~
Unit tests for VERTO. None of them needs torch, the model or the Spider data.
- SQL Executor & Normalization, including its read-only, ATTACH, time and row limits
- Schema Linker & Formatter
- Error-recovery loop
- Pydantic API Models and the FastAPI routes (with a fake engine)
- HTTP API Client
- CLI
"""

import inspect
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from verto import cli
from verto.api.client import VERTOClient
from verto.api.models import DatabasesResponse, HealthResponse, QueryRequest, QueryResponse
from verto.cli import build_parser
from verto.engine.executor import execute_sql, normalize_sql
from verto.engine.inference import build_prompt, run_with_recovery
from verto.engine.schema import load_and_format_schemas, load_tables_raw, schema_link


@pytest.fixture
def singer_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE singer (id INT, name TEXT)")
    cur.execute("INSERT INTO singer VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()
    conn.close()
    return db_path


# ── Executor Tests ────────────────────────────────────────────────────────────

def test_normalize_sql():
    raw_sql = "  SELECT   *  FROM   singers WHERE  age > 30 ; "
    expected = "select * from singers where age > 30 ;"
    assert normalize_sql(raw_sql) == expected


def test_execute_sql_success(singer_db: Path):
    res = execute_sql("SELECT name FROM singer ORDER BY id", singer_db)
    assert res == (["name"], [("Alice",), ("Bob",)])


def test_execute_sql_error(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    conn.close()

    res = execute_sql("SELECT * FROM non_existent_table", db_path)
    assert isinstance(res, str)
    assert res.startswith("Error:")


def test_execute_sql_missing_db(tmp_path: Path):
    missing_path = tmp_path / "non_existent.sqlite"
    res = execute_sql("SELECT 1", missing_path)
    assert isinstance(res, str)
    assert "Error:" in res


def test_execute_sql_is_read_only(singer_db: Path):
    for sql in ("DROP TABLE singer", "DELETE FROM singer", "INSERT INTO singer VALUES (3, 'Cy')"):
        assert execute_sql(sql, singer_db).startswith("Error:")
    assert execute_sql("SELECT count(*) FROM singer", singer_db) == (["count(*)"], [(2,)])


def test_execute_sql_cannot_attach_other_files(singer_db: Path, tmp_path: Path):
    other = tmp_path / "other.sqlite"
    assert execute_sql(f"ATTACH DATABASE '{other}' AS other", singer_db).startswith("Error:")
    assert not other.exists()


def test_execute_sql_stops_runaway_queries(singer_db: Path):
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    started = time.monotonic()
    res = execute_sql(endless, singer_db, timeout=0.2)
    assert res.startswith("Error:")
    assert time.monotonic() - started < 5


def test_execute_sql_caps_rows(singer_db: Path):
    many = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 50) SELECT x FROM c"
    columns, rows = execute_sql(many, singer_db, max_rows=10)
    assert len(rows) == 10


# ── Schema Linker Tests ───────────────────────────────────────────────────────

@pytest.fixture
def sample_tables_json(tmp_path: Path) -> Path:
    data = [
        {
            "db_id": "concert_singer",
            "table_names_original": ["stadium", "singer"],
            "column_names_original": [
                [-1, "*"],
                [0, "stadium_id"],
                [0, "location"],
                [0, "name"],
                [1, "singer_id"],
                [1, "name"],
                [1, "country"],
                [1, "age"],
            ],
            "column_types": ["text", "number", "text", "text", "number", "text", "text", "number"],
            "primary_keys": [1, 4],
            "foreign_keys": [],
        }
    ]
    path = tmp_path / "tables.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_and_format_schemas(sample_tables_json: Path):
    schemas = load_and_format_schemas(sample_tables_json)
    assert "concert_singer" in schemas
    schema_str = schemas["concert_singer"]
    assert "Table: stadium" in schema_str
    assert "Table: singer" in schema_str
    assert "[PK]" in schema_str


def test_schema_link_relevant_table(sample_tables_json: Path):
    schemas = load_and_format_schemas(sample_tables_json)
    raw_tables = load_tables_raw(sample_tables_json)

    linked = schema_link("How many singers are there?", "concert_singer", raw_tables, schemas, top_k_tables=1)
    assert "Table: singer" in linked
    assert "Table: stadium" not in linked
    assert linked in schemas["concert_singer"].split("\n")  # the same line as in the full schema


def test_schema_link_fallback(sample_tables_json: Path):
    schemas = load_and_format_schemas(sample_tables_json)
    raw_tables = load_tables_raw(sample_tables_json)

    linked = schema_link("xyz irrelevant query", "concert_singer", raw_tables, schemas, top_k_tables=1)
    assert "Table: stadium" in linked
    assert "Table: singer" in linked


# ── Error Recovery Tests ──────────────────────────────────────────────────────

SCHEMA = "Table: singer (id (number), name (text))"


def _candidates(*sqls):
    """A fake model: records each prompt and returns the same beam-search candidates."""
    prompts = []

    def generate(prompt: str) -> list[str]:
        prompts.append(prompt)
        return list(sqls)

    return generate, prompts


def test_recovery_runs_the_next_candidate(singer_db: Path):
    generate, prompts = _candidates("SELECT agez FROM singer", "SELECT name FROM singer ORDER BY id")
    res = run_with_recovery(generate, "Who sings?", SCHEMA, singer_db)
    assert res == {"sql": "SELECT name FROM singer ORDER BY id", "columns": ["name"],
                   "result": [["Alice"], ["Bob"]], "attempts": 2, "error": None}
    assert prompts == [build_prompt("Who sings?", SCHEMA)]  # one model call, however many tries


def test_no_fallback_when_error_recovery_is_off(singer_db: Path):
    generate, prompts = _candidates("SELECT agez FROM singer", "SELECT name FROM singer")
    res = run_with_recovery(generate, "Who sings?", SCHEMA, singer_db, use_error_recovery=False)
    assert res["attempts"] == 1
    assert res["error"] == "Error: no such column: agez"
    assert len(prompts) == 1


def test_when_nothing_runs_the_top_candidate_is_reported(singer_db: Path):
    generate, _ = _candidates("SELECT agez FROM singer", "SELECT name FROM nowhere")
    res = run_with_recovery(generate, "Who sings?", SCHEMA, singer_db)
    assert res["sql"] == "SELECT agez FROM singer"
    assert res["error"] == "Error: no such column: agez"
    assert res["attempts"] == 2


def test_duplicate_candidates_run_once(singer_db: Path):
    generate, _ = _candidates("SELECT agez FROM singer", "SELECT agez FROM singer", "SELECT name FROM singer")
    assert run_with_recovery(generate, "Who sings?", SCHEMA, singer_db)["attempts"] == 2


def test_less_than_survives_t5s_vocabulary():
    model = pytest.importorskip("verto.engine.model")  # needs torch
    sql = "SELECT name FROM singer WHERE age <= 30 AND id < 5"
    assert "<" not in model.to_t5_text(sql)
    assert model.from_t5_text(model.to_t5_text(sql)) == sql


# ── Pydantic Model Tests ──────────────────────────────────────────────────────

def test_pydantic_query_models():
    req = QueryRequest(question="List all singers", db_id="concert_singer")
    assert req.question == "List all singers"
    assert req.use_schema_linking is True

    resp = QueryResponse(sql="SELECT * FROM singer", columns=["id", "name"], result=[[1, "Alice"]], attempts=1)
    assert resp.sql == "SELECT * FROM singer"
    assert resp.columns == ["id", "name"]
    assert resp.error is None

    health = HealthResponse(status="ok", device="cpu", checkpoint_loaded=True, database_count=5)
    assert health.database_count == 5

    dbs = DatabasesResponse(databases=["concert_singer", "pets_1"], count=2)
    assert dbs.count == 2


# ── API Route Tests ───────────────────────────────────────────────────────────

class FakeEngine:
    device = "cpu"

    def list_databases(self):
        return ["concert_singer"]

    def query(self, question, db_id, use_schema_linking=True, use_error_recovery=True):
        return {"sql": "SELECT name FROM singer", "columns": ["name"], "result": [["Alice"]],
                "attempts": 1, "error": None}


@pytest.fixture
def api(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from verto.api import main

    monkeypatch.setattr(main, "_engine", FakeEngine())
    return TestClient(main.app)  # without `with`, the lifespan that loads the real model doesn't run


def test_api_query_returns_columns(api):
    res = api.post("/api/query", json={"question": "Who sings?", "db_id": "concert_singer"})
    assert res.status_code == 200
    assert res.json()["columns"] == ["name"]
    assert res.json()["result"] == [["Alice"]]


def test_api_rejects_overlong_questions(api):
    res = api.post("/api/query", json={"question": "x" * 1001, "db_id": "concert_singer"})
    assert res.status_code == 422


def test_api_health_and_databases(api):
    assert api.get("/api/health").json()["database_count"] == 1
    assert api.get("/api/databases").json() == {"databases": ["concert_singer"], "count": 1}


def test_api_sends_no_cors_headers(api):
    res = api.get("/api/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in res.headers


def test_query_route_runs_off_the_event_loop():
    pytest.importorskip("fastapi")
    from verto.api import main

    # FastAPI runs plain functions in a worker thread; an async one would block every other request.
    assert not inspect.iscoroutinefunction(main.query)


# ── HTTP Client Tests ─────────────────────────────────────────────────────────

class _SlowServer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):  # slower than the client's timeout
        time.sleep(1)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_GET(self):  # the server is up but the engine isn't
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"detail": "Engine not ready."}')


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # the client hangs up on the slow reply by design


@pytest.fixture
def slow_server():
    server = _QuietServer(("127.0.0.1", 0), _SlowServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_client_reports_a_timeout_instead_of_raising(slow_server):
    res = VERTOClient(slow_server, timeout=0.2).query("Who sings?", "concert_singer")
    assert res["sql"] is None
    assert res["error"].startswith("No answer")


def test_client_reports_http_errors_as_such(slow_server):
    with pytest.raises(RuntimeError, match="503: Engine not ready"):
        VERTOClient(slow_server).health()


# ── CLI Tests ─────────────────────────────────────────────────────────────────

def test_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--db", "concert_singer", "How many singers?"])
    assert args.db == "concert_singer"
    assert args.question == "How many singers?"
    assert args.local is False

    args_local = parser.parse_args(["--local", "--health"])
    assert args_local.local is True
    assert args_local.health is True


def _no_model(*args, **kwargs):
    pytest.fail("the CLI loaded the model")


def test_bare_verto_prints_help_without_loading_the_model(monkeypatch, capsys, sample_tables_json):
    import verto.engine.inference as inference

    monkeypatch.setattr(inference, "NLDIEngine", _no_model)
    monkeypatch.setattr("sys.argv", ["verto", "--spider-dir", str(sample_tables_json.parent)])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert "usage: verto" in capsys.readouterr().out


def test_local_list_dbs_reads_tables_json_without_the_model(monkeypatch, capsys, sample_tables_json):
    import verto.engine.inference as inference

    monkeypatch.setattr(inference, "NLDIEngine", _no_model)
    monkeypatch.setattr("sys.argv", ["verto", "--local", "--list-dbs", "--spider-dir", str(sample_tables_json.parent)])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert "concert_singer" in capsys.readouterr().out


def test_local_flag_never_falls_back_to_a_server(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(VERTOClient, "query", lambda *a, **k: pytest.fail("fell back to the server"))
    monkeypatch.setattr("sys.argv", ["verto", "--local", "--spider-dir", str(tmp_path),
                                     "-s", "http://127.0.0.1:9", "--db", "concert_singer", "Who sings?"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert "--local needs the Spider data" in capsys.readouterr().err
