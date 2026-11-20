"""
inference package — Clinical Evidence Intelligence System (CEIS)
"""

from inference.schemas import (
    CEISResult,
    SufficiencyResult,
    FlaggedPair,
    ContradictionMatrix,
)
from inference.highlighter import (
    extract_supporting_spans,
    extract_contradicting_spans,
    highlight_html,
)
from inference.chunker import PassageChunker, aggregate_chunk_results
from inference.contradiction import ContradictionDetector
from inference.pipeline import CEISPipeline, ClinicalEvidencePipeline

__all__ = [
    "CEISPipeline",
    "ClinicalEvidencePipeline",
    "CEISResult",
    "SufficiencyResult",
    "FlaggedPair",
    "ContradictionMatrix",
    "PassageChunker",
    "aggregate_chunk_results",
    "extract_supporting_spans",
    "extract_contradicting_spans",
    "highlight_html",
    "ContradictionDetector",
]
