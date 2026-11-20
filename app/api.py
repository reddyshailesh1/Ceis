"""
CEIS FastAPI backend.

Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health
GET  /api/v1/models
POST /api/v1/verify
POST /api/v1/scan
POST /api/v1/batch
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `inference.pipeline` is importable
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.schemas import (
    BatchVerifyRequest,
    BatchVerifyResponse,
    HealthResponse,
    ModelsResponse,
    ScanRequest,
    ScanResponse,
    VerifyRequest,
    VerifyResponse,
)

logger = logging.getLogger("ceis.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Model registry
# Key = model id/path string, Value = loaded pipeline object (or None)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "pavanmantha/mednli_laya"
_KNOWN_MODELS: list[str] = [
    "pavanmantha/mednli_laya",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "roberta-large-mnli",
]

# Discover any local `runs/` checkpoints
_RUNS_DIR = os.path.join(_ROOT, "runs")
if os.path.isdir(_RUNS_DIR):
    for _entry in os.scandir(_RUNS_DIR):
        if _entry.is_dir():
            _KNOWN_MODELS.append(f"runs/{_entry.name}")

_model_registry: dict[str, Any] = {}   # model_id -> pipeline object


def _load_pipeline(model_id: str):
    """Load (or return cached) CEIS inference pipeline for the given model."""
    if model_id in _model_registry and _model_registry[model_id] is not None:
        return _model_registry[model_id]

    logger.info("Loading model: %s", model_id)
    try:
        # Resolve local runs/ paths relative to project root
        if model_id.startswith("runs/"):
            model_path = os.path.join(_ROOT, model_id)
        else:
            model_path = model_id

        from inference.pipeline import CEISPipeline  # noqa: PLC0415
        pipeline = CEISPipeline(model_path)
        _model_registry[model_id] = pipeline
        logger.info("Model loaded: %s", model_id)
        return pipeline
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_id, exc)
        raise RuntimeError(f"Cannot load model '{model_id}': {exc}") from exc


def _get_device() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Lifespan — load default model at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CEIS API starting — loading default model '%s'…", _DEFAULT_MODEL)
    try:
        _load_pipeline(_DEFAULT_MODEL)
    except Exception as exc:
        logger.warning("Default model failed to load at startup (%s). It will be retried on first request.", exc)
    yield
    logger.info("CEIS API shutting down.")


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Clinical Evidence Intelligence System (CEIS)",
    description="REST API for clinical claim verification powered by layered NLI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Streamlit dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result_to_verify_response(result) -> VerifyResponse:
    """Convert a CEISResult (dataclass / dict) to a VerifyResponse."""
    if hasattr(result, "__dict__"):
        r = result.__dict__
    else:
        r = dict(result)

    return VerifyResponse(
        verdict=r.get("verdict", "unsupported"),
        confidence=float(r.get("confidence", 0.0)),
        probabilities={k: float(v) for k, v in r.get("probabilities", {}).items()},
        explicitly_mentioned=bool(r.get("explicitly_mentioned", False)),
        needs_more_evidence=bool(r.get("needs_more_evidence", True)),
        evidence_sufficiency=float(r.get("evidence_sufficiency", 0.0)),
        inconsistency_severity=float(r.get("inconsistency_severity", 0.0)),
        human_review_recommended=bool(r.get("human_review_recommended", False)),
        reasoning=str(r.get("reasoning", "")),
        supporting_spans=list(r.get("supporting_spans", [])),
        contradicting_spans=list(r.get("contradicting_spans", [])),
        highlighted_evidence_html=str(r.get("highlighted_evidence_html", "")),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Return API health status and loaded model info."""
    loaded = [k for k, v in _model_registry.items() if v is not None]
    return HealthResponse(
        status="ok" if loaded else "degraded",
        loaded_models=loaded,
        device=_get_device(),
    )


@app.get("/api/v1/models", response_model=ModelsResponse, tags=["Models"])
async def list_models():
    """List known and currently-loaded models."""
    loaded = [k for k, v in _model_registry.items() if v is not None]
    return ModelsResponse(
        available=_KNOWN_MODELS,
        loaded=loaded,
        default=_DEFAULT_MODEL,
    )


@app.post("/api/v1/verify", response_model=VerifyResponse, tags=["Inference"])
async def verify(req: VerifyRequest):
    """
    Verify a single (claim, evidence) pair.

    Returns a structured verdict with confidence, decision outputs, highlighted
    evidence HTML, and a natural-language reasoning summary.
    """
    try:
        pipeline = _load_pipeline(req.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = pipeline.verify(claim=req.claim, evidence=req.evidence)
    except Exception as exc:
        logger.exception("Inference error for model %s", req.model)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return _result_to_verify_response(result)


@app.post("/api/v1/scan", response_model=ScanResponse, tags=["Inference"])
async def scan(req: ScanRequest):
    """
    Scan a list of clinical statements for pairwise contradictions.

    Returns an NxN verdict matrix, flagged contradictory pairs, and a
    human-readable summary.
    """
    if len(req.statements) < 2:
        raise HTTPException(
            status_code=422,
            detail="At least 2 statements are required for a contradiction scan.",
        )
    if len(req.statements) > 50:
        raise HTTPException(
            status_code=422,
            detail="At most 50 statements are supported per scan request.",
        )

    try:
        pipeline = _load_pipeline(req.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    n = len(req.statements)
    matrix: list[list[str]] = [["self" if i == j else "" for j in range(n)] for i in range(n)]
    flagged_pairs: list[dict] = []

    try:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                result = pipeline.verify(claim=req.statements[i], evidence=req.statements[j])
                r = _result_to_verify_response(result)
                matrix[i][j] = r.verdict
                if r.verdict == "contradicted":
                    flagged_pairs.append(
                        {
                            "claim_idx": i,
                            "evidence_idx": j,
                            "claim": req.statements[i],
                            "evidence": req.statements[j],
                            "confidence": r.confidence,
                            "inconsistency_severity": r.inconsistency_severity,
                        }
                    )
    except Exception as exc:
        logger.exception("Scan inference error")
        raise HTTPException(status_code=500, detail=f"Scan inference failed: {exc}") from exc

    n_contradictions = len(flagged_pairs)
    summary = (
        f"Scanned {n} statements ({n * (n - 1)} ordered pairs). "
        f"Found {n_contradictions} contradictory pair(s)."
    )
    if n_contradictions == 0:
        summary += " No contradictions detected."

    return ScanResponse(matrix=matrix, flagged_pairs=flagged_pairs, summary=summary)


@app.post("/api/v1/batch", response_model=BatchVerifyResponse, tags=["Inference"])
async def batch_verify(req: BatchVerifyRequest):
    """
    Verify a batch of (claim, evidence) pairs.

    Items that fail individually are recorded in the `errors` list; the rest
    are returned normally. The response always has `total` == len(pairs).
    """
    if not req.pairs:
        raise HTTPException(status_code=422, detail="pairs list must not be empty.")
    if len(req.pairs) > 200:
        raise HTTPException(status_code=422, detail="At most 200 pairs per batch request.")

    for idx, pair in enumerate(req.pairs):
        if "claim" not in pair or "evidence" not in pair:
            raise HTTPException(
                status_code=422,
                detail=f"Item {idx} is missing 'claim' or 'evidence' key.",
            )

    try:
        pipeline = _load_pipeline(req.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: list[VerifyResponse] = []
    errors: list[dict] = []

    for idx, pair in enumerate(req.pairs):
        try:
            result = pipeline.verify(claim=pair["claim"], evidence=pair["evidence"])
            results.append(_result_to_verify_response(result))
        except Exception as exc:
            logger.warning("Batch item %d failed: %s", idx, exc)
            errors.append({"index": idx, "message": str(exc)})
            # Insert a placeholder so indices stay aligned
            results.append(
                VerifyResponse(
                    verdict="error",
                    confidence=0.0,
                    probabilities={},
                    explicitly_mentioned=False,
                    needs_more_evidence=True,
                    evidence_sufficiency=0.0,
                    inconsistency_severity=0.0,
                    human_review_recommended=True,
                    reasoning=f"Inference error: {exc}",
                    supporting_spans=[],
                    contradicting_spans=[],
                    highlighted_evidence_html="",
                )
            )

    return BatchVerifyResponse(results=results, total=len(req.pairs), errors=errors)
