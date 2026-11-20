"""
schemas.py — All dataclasses for the CEIS inference pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class CEISResult:
    """Full structured result for a single (claim, evidence) verification."""

    # Core verdict
    verdict: str                          # "supported" | "unsupported" | "contradicted"
    confidence: float                     # probability of the predicted verdict (0-1)
    probabilities: Dict[str, float]       # {"supported": p, "unsupported": p, "contradicted": p}

    # Ancillary boolean decisions
    explicitly_mentioned: bool            # Does evidence explicitly mention the claim topic?
    needs_more_evidence: bool             # Is additional evidence needed?

    # Continuous scores
    evidence_sufficiency: float           # 0-1; how sufficient the evidence is
    inconsistency_severity: float         # 0-1; 0 when verdict != "contradicted"

    # Review flag
    human_review_recommended: bool        # True if confidence<0.7 OR contradicted OR severity>0.6

    # Textual outputs
    reasoning: str                        # Templated explanation (never hallucinated)
    supporting_spans: List[str]           # Sentences from evidence supporting the claim
    contradicting_spans: List[str]        # Sentences from evidence contradicting the claim

    # Optional raw inputs stored for downstream use
    claim: str = ""
    evidence: str = ""
    highlighted_evidence_html: str = ""   # HTML formatted evidence with span marks


@dataclass
class SufficiencyResult:
    """Result for one claim in an analyze_sufficiency call."""
    claim: str
    evidence_sufficiency: float
    needs_more_evidence: bool
    verdict: str
    confidence: float
    full_result: Optional[CEISResult] = None


@dataclass
class FlaggedPair:
    """A pair of statements flagged as contradictory."""
    index_a: int
    index_b: int
    statement_a: str
    statement_b: str
    verdict_ab: str                       # verdict when A is claim, B is evidence
    verdict_ba: str                       # verdict when B is claim, A is evidence
    confidence_ab: float
    confidence_ba: float


@dataclass
class ContradictionMatrix:
    """Full NxN pairwise contradiction analysis."""
    statements: List[str]
    verdicts: List[List[str]]             # NxN matrix of verdict strings
    confidences: List[List[float]]        # NxN matrix of confidence floats
    flagged_pairs: List[FlaggedPair]      # Only contradicted pairs
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.statements)
