"""
Pydantic v2 request/response schemas for the CEIS API.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    claim: str = Field(..., description="The clinical claim to verify")
    evidence: str = Field(..., description="The evidence passage to verify against")
    model: str = Field(
        "pavanmantha/mednli_laya",
        description="HuggingFace model ID or local path to use for inference",
    )


class ScanRequest(BaseModel):
    statements: list[str] = Field(
        ..., description="List of clinical statements to scan for contradictions"
    )
    model: str = Field(
        "pavanmantha/mednli_laya",
        description="HuggingFace model ID or local path",
    )


class BatchVerifyRequest(BaseModel):
    pairs: list[dict] = Field(
        ..., description="List of dicts with 'claim' and 'evidence' keys"
    )
    model: str = Field(
        "pavanmantha/mednli_laya",
        description="HuggingFace model ID or local path",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class VerifyResponse(BaseModel):
    verdict: str = Field(..., description="supported | unsupported | contradicted")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(
        ..., description="Per-class probabilities keyed by label name"
    )
    explicitly_mentioned: bool = Field(
        ..., description="Whether the claim topic is explicitly mentioned in the evidence"
    )
    needs_more_evidence: bool = Field(
        ..., description="Whether additional evidence is needed to reach a conclusion"
    )
    evidence_sufficiency: float = Field(..., ge=0.0, le=1.0)
    inconsistency_severity: float = Field(..., ge=0.0, le=1.0)
    human_review_recommended: bool = Field(
        ..., description="True when low confidence or high severity warrants expert review"
    )
    reasoning: str = Field(..., description="Natural-language reasoning summary")
    supporting_spans: list[str] = Field(
        default_factory=list,
        description="Evidence spans that support the claim",
    )
    contradicting_spans: list[str] = Field(
        default_factory=list,
        description="Evidence spans that contradict the claim",
    )
    highlighted_evidence_html: str = Field(
        ..., description="Evidence text with HTML highlights for supporting/contradicting spans"
    )


class ScanResponse(BaseModel):
    matrix: list[list[str]] = Field(
        ..., description="NxN matrix of verdict labels (diagonal is 'self')"
    )
    flagged_pairs: list[dict] = Field(
        ..., description="Pairs where verdict == 'contradicted'"
    )
    summary: str = Field(..., description="Human-readable scan summary")


class BatchVerifyResponse(BaseModel):
    results: list[VerifyResponse] = Field(
        ..., description="Ordered list of verification results for each input pair"
    )
    total: int
    errors: list[dict] = Field(
        default_factory=list,
        description="Per-item errors if any item failed (index + message)",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded | error")
    loaded_models: list[str] = Field(
        ..., description="Model IDs currently loaded into memory"
    )
    device: str = Field(..., description="torch device string, e.g. 'cuda:0' or 'cpu'")


class ModelsResponse(BaseModel):
    available: list[str] = Field(..., description="Models known to the registry")
    loaded: list[str] = Field(..., description="Models currently in GPU/CPU memory")
    default: str
