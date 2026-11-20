"""
highlighter.py — Evidence span extraction and HTML highlighting.

No attention extraction — uses TF-IDF cosine similarity for supporting spans
and negation + keyword matching for contradicting spans.
"""
from __future__ import annotations

import re
import html
from typing import List

# ---------------------------------------------------------------------------
# Sentence tokenisation
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "without", "absent", "denied", "denies",
    "negative", "negatives", "none", "nor", "cannot", "doesn't", "isn't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't", "wouldn't",
    "didn't", "don't", "unremarkable", "normal", "within normal limits",
})


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences; fall back to the whole text if too short."""
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return sents if sents else [text.strip()]


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens (alphanumeric only)."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (pure-Python, no sklearn dependency)
# ---------------------------------------------------------------------------

def _tfidf_cosine(query_tokens: List[str], candidate_tokens: List[str],
                  corpus_idf: dict[str, float]) -> float:
    """Compute TF-IDF cosine similarity between query and candidate token lists."""
    if not query_tokens or not candidate_tokens:
        return 0.0

    def tfidf_vec(tokens: List[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = max(len(tokens), 1)
        return {t: (c / n) * corpus_idf.get(t, 1.0) for t, c in tf.items()}

    q_vec = tfidf_vec(query_tokens)
    c_vec = tfidf_vec(candidate_tokens)

    shared = set(q_vec) & set(c_vec)
    dot = sum(q_vec[t] * c_vec[t] for t in shared)
    norm_q = sum(v * v for v in q_vec.values()) ** 0.5
    norm_c = sum(v * v for v in c_vec.values()) ** 0.5
    if norm_q == 0 or norm_c == 0:
        return 0.0
    return dot / (norm_q * norm_c)


def _build_idf(sentences: List[List[str]]) -> dict[str, float]:
    """Compute IDF from a list of tokenised sentences."""
    import math
    n = len(sentences)
    df: dict[str, int] = {}
    for tokens in sentences:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_supporting_spans(claim: str, evidence: str, max_spans: int = 3) -> List[str]:
    """Find sentences in evidence most lexically similar to the claim (TF-IDF cosine).

    Parameters
    ----------
    claim:     The claim string.
    evidence:  The evidence passage.
    max_spans: Maximum number of sentences to return.

    Returns
    -------
    List of sentence strings, ordered by decreasing similarity.
    """
    sentences = _split_sentences(evidence)
    if not sentences:
        return []

    tokenised = [_tokenize(s) for s in sentences]
    idf = _build_idf(tokenised)
    claim_tokens = _tokenize(claim)

    scored = [
        (i, _tfidf_cosine(claim_tokens, tok, idf))
        for i, tok in enumerate(tokenised)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Only include sentences with non-zero similarity
    top = [(i, score) for i, score in scored if score > 0.0][:max_spans]
    return [sentences[i] for i, _ in top]


def extract_contradicting_spans(claim: str, evidence: str, verdict: str,
                                 max_spans: int = 2) -> List[str]:
    """Find sentences that negate claim keywords (only called when verdict==contradicted).

    Strategy:
    1. Find sentences containing claim keywords.
    2. Among those, prefer sentences containing negation words.
    3. Fall back to top TF-IDF matches if no negation found.

    Parameters
    ----------
    claim:    The claim string.
    evidence: The evidence passage.
    verdict:  Should be "contradicted"; function is a no-op for other verdicts.
    max_spans: Maximum sentences to return.

    Returns
    -------
    List of sentence strings.
    """
    if verdict != "contradicted":
        return []

    sentences = _split_sentences(evidence)
    if not sentences:
        return []

    claim_tokens = set(_tokenize(claim))

    negation_hits: List[str] = []
    keyword_hits: List[str] = []

    for sent in sentences:
        sent_tokens = set(_tokenize(sent))
        has_keyword = bool(claim_tokens & sent_tokens)
        has_negation = bool(sent_tokens & _NEGATION_WORDS)

        if has_keyword and has_negation:
            negation_hits.append(sent)
        elif has_keyword:
            keyword_hits.append(sent)

    if negation_hits:
        return negation_hits[:max_spans]
    if keyword_hits:
        return keyword_hits[:max_spans]

    # Last resort: TF-IDF top match
    return extract_supporting_spans(claim, evidence, max_spans)


def highlight_html(evidence: str, supporting: List[str], contradicting: List[str]) -> str:
    """Return HTML with highlighted supporting and contradicting spans.

    Supporting spans: <mark class='support'>...</mark>
    Contradicting spans: <mark class='contradict'>...</mark>

    Spans are matched by exact substring (after HTML-escaping). If a span
    appears in both lists, contradicting takes priority.

    Parameters
    ----------
    evidence:      The original evidence text.
    supporting:    List of supporting sentence strings.
    contradicting: List of contradicting sentence strings.

    Returns
    -------
    HTML string.
    """
    # Build sets for quick lookup; contradicting takes priority
    support_set = set(supporting)
    contradict_set = set(contradicting)

    # Remove from support anything that is also in contradict
    support_set -= contradict_set

    # Escape the full evidence for HTML
    escaped = html.escape(evidence)

    # Apply contradicting highlights first (higher priority)
    for span in sorted(contradict_set, key=len, reverse=True):
        esc_span = html.escape(span)
        if esc_span in escaped:
            escaped = escaped.replace(
                esc_span,
                f"<mark class='contradict'>{esc_span}</mark>",
                1,
            )

    # Apply supporting highlights
    for span in sorted(support_set, key=len, reverse=True):
        esc_span = html.escape(span)
        if esc_span in escaped:
            escaped = escaped.replace(
                esc_span,
                f"<mark class='support'>{esc_span}</mark>",
                1,
            )

    return f"<p>{escaped}</p>"
