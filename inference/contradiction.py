"""
contradiction.py — Multi-statement pairwise contradiction detection.

Uses ClinicalEvidencePipeline to perform an NxN scan: for every ordered pair
(A, B) of statements, A is treated as the claim and B as the evidence.
A pair is flagged when EITHER direction yields "contradicted".
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from inference.schemas import ContradictionMatrix, FlaggedPair

if TYPE_CHECKING:
    from inference.pipeline import ClinicalEvidencePipeline


class ContradictionDetector:
    """Detect pairwise contradictions among a list of clinical statements.

    Parameters
    ----------
    pipeline:
        An initialised ``ClinicalEvidencePipeline`` instance.
    """

    def __init__(self, pipeline: "ClinicalEvidencePipeline"):
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, statements: List[str]) -> ContradictionMatrix:
        """Run pairwise NxN contradiction scan.

        For every ordered pair (i, j) with i ≠ j, statement i is the *claim*
        and statement j is the *evidence*.  The diagonal (i == i) is filled
        with ``"self"`` and confidence ``1.0``.

        A pair (i, j) is flagged as contradicted when statement i treats
        statement j as evidence and receives a ``"contradicted"`` verdict, OR
        vice-versa.

        Parameters
        ----------
        statements:
            List of clinical statements to compare.  At least 2 required.

        Returns
        -------
        ContradictionMatrix
        """
        n = len(statements)
        if n < 2:
            raise ValueError("detect() requires at least 2 statements")

        # Build all off-diagonal pairs as (claim, evidence) tuples
        pairs: List[tuple[str, str]] = []
        pair_indices: List[tuple[int, int]] = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pairs.append((statements[i], statements[j]))
                pair_indices.append((i, j))

        # Batch verify all pairs at once
        results = self._pipeline.batch_verify(pairs)

        # Populate NxN matrices
        verdicts: List[List[str]] = [["self"] * n for _ in range(n)]
        confidences: List[List[float]] = [[1.0] * n for _ in range(n)]

        result_map: dict[tuple[int, int], object] = {}
        for (i, j), res in zip(pair_indices, results):
            verdicts[i][j] = res.verdict
            confidences[i][j] = res.confidence
            result_map[(i, j)] = res

        # Collect flagged pairs (symmetric: flag when either direction is contradicted)
        flagged_pairs: List[FlaggedPair] = []
        seen: set[frozenset[int]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                v_ij = verdicts[i][j]
                v_ji = verdicts[j][i]
                if v_ij == "contradicted" or v_ji == "contradicted":
                    key = frozenset({i, j})
                    if key not in seen:
                        seen.add(key)
                        flagged_pairs.append(
                            FlaggedPair(
                                index_a=i,
                                index_b=j,
                                statement_a=statements[i],
                                statement_b=statements[j],
                                verdict_ab=v_ij,
                                verdict_ba=v_ji,
                                confidence_ab=confidences[i][j],
                                confidence_ba=confidences[j][i],
                            )
                        )

        return ContradictionMatrix(
            statements=list(statements),
            verdicts=verdicts,
            confidences=confidences,
            flagged_pairs=flagged_pairs,
        )

    def summarize(self, matrix: ContradictionMatrix) -> str:
        """Return a human-readable plain-text summary of detected contradictions.

        Parameters
        ----------
        matrix:
            A ``ContradictionMatrix`` produced by :meth:`detect`.

        Returns
        -------
        Multi-line string.
        """
        n = matrix.n
        total_pairs = n * (n - 1) // 2
        n_flagged = len(matrix.flagged_pairs)

        lines: List[str] = [
            f"Contradiction scan over {n} statements ({total_pairs} unique pairs):",
            f"  Flagged pairs: {n_flagged} / {total_pairs}",
        ]

        if n_flagged == 0:
            lines.append("  No contradictions detected.")
            return "\n".join(lines)

        lines.append("")
        lines.append("Flagged contradictions:")

        for idx, pair in enumerate(matrix.flagged_pairs, start=1):
            a_short = _truncate(pair.statement_a, 80)
            b_short = _truncate(pair.statement_b, 80)
            conf_ab = round(pair.confidence_ab * 100, 1)
            conf_ba = round(pair.confidence_ba * 100, 1)

            direction_parts: List[str] = []
            if pair.verdict_ab == "contradicted":
                direction_parts.append(
                    f"[{pair.index_a}]→[{pair.index_b}] contradicted ({conf_ab}% conf)"
                )
            if pair.verdict_ba == "contradicted":
                direction_parts.append(
                    f"[{pair.index_b}]→[{pair.index_a}] contradicted ({conf_ba}% conf)"
                )

            lines.append(f"  {idx}. Statements [{pair.index_a}] & [{pair.index_b}]")
            lines.append(f"       [{pair.index_a}]: {a_short}")
            lines.append(f"       [{pair.index_b}]: {b_short}")
            lines.append(f"       Directions: {'; '.join(direction_parts)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* chars, appending '…' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
