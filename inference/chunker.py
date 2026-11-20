"""
chunker.py — Long-passage handler for the CEIS inference pipeline.

Splits long evidence texts into overlapping chunks so each chunk fits within
the model's token budget. Chunk results are then aggregated via majority vote
(verdict) and mean (continuous scores).
"""
from __future__ import annotations

import re
from typing import List, Optional

from inference.schemas import CEISResult

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TOKENS = 400
_DEFAULT_OVERLAP = 50
_WORD_RE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Tokeniser shim
# ---------------------------------------------------------------------------

class _WhitespaceTokenizer:
    """Minimal tokeniser used when no HF tokeniser is available.

    Splits on whitespace.  One 'token' ≈ one word.
    """

    @staticmethod
    def encode(text: str) -> List[str]:
        return _WORD_RE.findall(text)

    @staticmethod
    def decode(tokens: List[str]) -> str:
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# PassageChunker
# ---------------------------------------------------------------------------

class PassageChunker:
    """Split a long text into overlapping chunks that fit within ``max_tokens``.

    Parameters
    ----------
    max_tokens:
        Maximum number of tokens per chunk.  Default: 400.
    overlap:
        Number of tokens to repeat at the start of the next chunk to preserve
        context across chunk boundaries.  Default: 50.
    tokenizer:
        A HuggingFace tokeniser object (or any object with ``encode`` /
        ``decode`` methods that work on strings).  When *None*, falls back to
        a simple whitespace tokeniser.
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        overlap: int = _DEFAULT_OVERLAP,
        tokenizer: Optional[object] = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.overlap = overlap
        self._tok = tokenizer or _WhitespaceTokenizer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list:
        """Encode text → token list (works for HF tokenisers and our shim)."""
        enc = self._tok.encode(text)
        # HF tokenisers return an object with .ids; plain lists are fine too.
        if hasattr(enc, "ids"):
            return enc.ids
        return enc

    def _decode(self, tokens: list) -> str:
        """Decode token list → string."""
        if hasattr(self._tok, "decode"):
            return self._tok.decode(tokens)
        # Fallback: join as strings
        return " ".join(str(t) for t in tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks.

        Returns
        -------
        List of chunk strings.  If the text fits in one chunk the list has
        exactly one element (the original text, stripped).
        """
        text = text.strip()
        if not text:
            return []

        tokens = self._encode(text)
        if len(tokens) <= self.max_tokens:
            return [text]

        step = self.max_tokens - self.overlap
        if step <= 0:
            step = 1

        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end == len(tokens):
                break
            start += step

        return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_chunk_results(results: List[CEISResult]) -> CEISResult:
    """Aggregate multiple chunk results into a single CEISResult.

    Strategy
    --------
    - **verdict**: majority vote; ties broken by priority:
      contradicted > supported > unsupported.
    - **confidence**: mean of per-chunk confidences for the winning verdict.
    - **probabilities**: element-wise mean across chunks.
    - **explicitly_mentioned**: True if any chunk says True.
    - **needs_more_evidence**: True if any chunk says True (conservative).
    - **evidence_sufficiency**: mean across chunks.
    - **inconsistency_severity**: max across chunks (worst case).
    - **human_review_recommended**: recomputed from aggregated values.
    - **reasoning**: constructed from aggregated values (not from any single
      chunk to avoid partial information).
    - **supporting_spans / contradicting_spans**: union, deduplicated,
      preserving order.
    - **claim / evidence**: taken from the first result.

    Parameters
    ----------
    results:
        Non-empty list of CEISResult objects from individual chunks.

    Returns
    -------
    A single aggregated CEISResult.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("aggregate_chunk_results requires at least one result")
    if len(results) == 1:
        return results[0]

    # --- Verdict: majority vote with tie-breaking priority ---
    _PRIORITY = {"contradicted": 0, "supported": 1, "unsupported": 2}
    vote_counts: dict[str, int] = {}
    for r in results:
        vote_counts[r.verdict] = vote_counts.get(r.verdict, 0) + 1
    max_votes = max(vote_counts.values())
    candidates = [v for v, c in vote_counts.items() if c == max_votes]
    verdict = min(candidates, key=lambda v: _PRIORITY.get(v, 99))

    # --- Confidence: mean of chunks that voted for the winning verdict ---
    winning_confs = [r.confidence for r in results if r.verdict == verdict]
    confidence = sum(winning_confs) / len(winning_confs)

    # --- Probabilities: element-wise mean ---
    all_labels = ["supported", "unsupported", "contradicted"]
    probabilities: dict[str, float] = {}
    for lbl in all_labels:
        vals = [r.probabilities.get(lbl, 0.0) for r in results]
        probabilities[lbl] = sum(vals) / len(vals)
    # Renormalise to sum to 1
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v / total for k, v in probabilities.items()}

    # --- Boolean flags ---
    explicitly_mentioned = any(r.explicitly_mentioned for r in results)
    needs_more_evidence = any(r.needs_more_evidence for r in results)

    # --- Continuous scores ---
    evidence_sufficiency = sum(r.evidence_sufficiency for r in results) / len(results)
    inconsistency_severity = max(r.inconsistency_severity for r in results)

    # --- Human review ---
    human_review_recommended = (
        confidence < 0.7
        or verdict == "contradicted"
        or inconsistency_severity > 0.6
    )

    # --- Spans: union, deduplicated, order preserved ---
    def _dedup(spans_lists) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for lst in spans_lists:
            for s in lst:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out

    supporting_spans = _dedup(r.supporting_spans for r in results)
    contradicting_spans = _dedup(r.contradicting_spans for r in results)

    # --- Reasoning ---
    mention_phrase = (
        "explicitly mentions" if explicitly_mentioned else "does not explicitly mention"
    )
    more_phrase = "is" if needs_more_evidence else "is not"
    suf_pct = round(evidence_sufficiency * 100, 1)
    conf_pct = round(confidence * 100, 1)
    reasoning = (
        f"The evidence {mention_phrase} the claim topic. "
        f"The model classified this as {verdict} with {conf_pct}% confidence "
        f"(aggregated over {len(results)} evidence chunks). "
        f"Additional evidence {more_phrase} needed. "
        f"Evidence sufficiency score: {suf_pct}%."
    )
    if verdict == "contradicted":
        sev_pct = round(inconsistency_severity * 100, 1)
        reasoning += f" Inconsistency severity: {sev_pct}%."

    return CEISResult(
        verdict=verdict,
        confidence=confidence,
        probabilities=probabilities,
        explicitly_mentioned=explicitly_mentioned,
        needs_more_evidence=needs_more_evidence,
        evidence_sufficiency=evidence_sufficiency,
        inconsistency_severity=inconsistency_severity,
        human_review_recommended=human_review_recommended,
        reasoning=reasoning,
        supporting_spans=supporting_spans,
        contradicting_spans=contradicting_spans,
        claim=results[0].claim,
        evidence=results[0].evidence,
    )
