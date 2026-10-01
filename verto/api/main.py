"""
verto/api/main.py
~~~~~~~~~~~~~~~~~
FastAPI application for VERTO.

Start with:
    uvicorn verto.api.main:app --reload --port 8000

Endpoints
---------
GET  /api/health        — liveness check + model info
GET  /api/databases     — list available Spider database IDs
POST /api/query         — translate a question to SQL and execute it
GET  /                  — serve the frontend (web/)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from verto import config as cfg
from verto.engine.inference import NLDIEngine

from .models import DatabasesResponse, HealthResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

# Project root: verto/api/main.py → parent.parent.parent = VERTO/
_WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

# ── Engine singleton ─────────────────────────────────────────────────────────

_engine: NLDIEngine | None = None
_checkpoint_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _checkpoint_loaded

    logger.info("Starting VERTO server …")
    _checkpoint_loaded = Path(cfg.CHECKPOINT_PATH).exists()

    _engine = NLDIEngine(
        checkpoint_path=cfg.CHECKPOINT_PATH,
        spider_dir=cfg.SPIDER_DIR,
        device=cfg.DEVICE,
    )
    logger.info("Engine ready — %d databases available.", len(_engine.list_databases()))

    yield

    logger.info("Shutting down VERTO server.")
    _engine = None


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VERTO — Natural Language Database Interface Engine",
    description=(
        "Fine-tuned T5+LoRA text-to-SQL engine. "
        "Translate plain-English questions into executable SQL queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# No CORS middleware: the console is served from this same origin, and the CLI is not a
# browser, so neither needs cross-origin access.

if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")



# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = _WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — confirms the model is loaded and ready."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    return HealthResponse(
        status="ok",
        device=str(_engine.device),
        checkpoint_loaded=_checkpoint_loaded,
        database_count=len(_engine.list_databases()),
    )


@app.get("/api/databases", response_model=DatabasesResponse, tags=["Databases"])
async def list_databases():
    """Return all Spider database IDs available for querying."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    dbs = _engine.list_databases()
    return DatabasesResponse(databases=dbs, count=len(dbs))


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
def query(body: QueryRequest):
    """Translate a plain-English question into SQL and execute it.

    A plain function, not ``async``: FastAPI runs it in a worker thread, so a query that
    takes seconds doesn't hold up other requests such as /api/health.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")

    result = _engine.query(
        question=body.question,
        db_id=body.db_id,
        use_schema_linking=body.use_schema_linking,
        use_error_recovery=body.use_error_recovery,
    )
    return QueryResponse(**result)
