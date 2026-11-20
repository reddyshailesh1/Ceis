"""
schema.py  —  Multi-question CEIS schema
-----------------------------------------
Extends ClaimCheckSchema to 5 questions fired per (claim, evidence) pair:

  1. verdict              (choice) : supported / unsupported / contradicted
  2. explicitly_mentioned (yesno)  : does evidence explicitly mention the topic?
  3. needs_more_evidence  (yesno)  : is additional evidence required?
  4. evidence_sufficiency (score)  : 0.0–1.0
  5. inconsistency_severity (score): 0.0–1.0

Verdict has 6 training wording variants (4 inherited + 2 new) plus a held-out
benchmark wording.  Every variant dictionary follows the laya schema contract:
  {"type": "choice|yesno|score", "instructions": str, "criteria": {...}}
"""

from __future__ import annotations

from typing import Optional


class CEISSchema:
    # ------------------------------------------------------------------
    # Question IDs
    # ------------------------------------------------------------------
    QID_VERDICT    = "verdict"
    QID_MENTIONED  = "explicitly_mentioned"
    QID_NEEDS_MORE = "needs_more_evidence"
    QID_SUFFICIENCY = "evidence_sufficiency"
    QID_SEVERITY   = "inconsistency_severity"

    # Canonical label order (must stay stable — indices are used as targets)
    LABELS = ("supported", "unsupported", "contradicted")

    # Mapping from NLI gold labels → CEIS labels
    FROM_NLI = {
        "entailment":   "supported",
        "neutral":      "unsupported",
        "contradiction": "contradicted",
    }

    # ------------------------------------------------------------------
    # Verdict (choice) variants — 6 total (4 from ClaimCheck + 2 new)
    # ------------------------------------------------------------------
    TRAIN_VARIANTS = [
        # Variant 0  (from original ClaimCheckSchema)
        {
            "instructions": "Check the claim against the evidence.",
            "criteria": {
                "supported":    "the evidence establishes that the claim is true",
                "unsupported":  "the evidence neither establishes nor rules out the claim",
                "contradicted": "the evidence rules out the claim",
            },
        },
        # Variant 1  (from original ClaimCheckSchema)
        {
            "instructions": "Given only the evidence, what is the status of the claim?",
            "criteria": {
                "supported":    "the claim follows from the evidence",
                "unsupported":  "the claim cannot be determined from the evidence",
                "contradicted": "the claim conflicts with the evidence",
            },
        },
        # Variant 2  (from original ClaimCheckSchema)
        {
            "instructions": "Is the claim backed by the clinical evidence provided?",
            "criteria": {
                "supported":    "yes, the evidence backs the claim",
                "unsupported":  "the evidence is not enough to decide",
                "contradicted": "no, the evidence shows the claim is false",
            },
        },
        # Variant 3  (from original ClaimCheckSchema)
        {
            "instructions": "Verify the statement using the passage.",
            "criteria": {
                "supported":    "the passage entails the statement",
                "unsupported":  "the passage is neutral about the statement",
                "contradicted": "the passage contradicts the statement",
            },
        },
        # Variant 4  — NEW: clinical framing
        {
            "instructions": (
                "A clinician wrote the following claim. "
                "Assess whether the medical evidence supports or refutes it."
            ),
            "criteria": {
                "supported":    (
                    "the clinical evidence directly supports the claim — "
                    "the finding, value, or diagnosis is confirmed"
                ),
                "unsupported":  (
                    "the clinical evidence is silent or inconclusive about the claim"
                ),
                "contradicted": (
                    "the clinical evidence directly refutes the claim — "
                    "the finding, value, or diagnosis is inconsistent"
                ),
            },
        },
        # Variant 5  — NEW: NLI framing
        {
            "instructions": (
                "Does the following medical passage entail, contradict, "
                "or remain neutral regarding the hypothesis?"
            ),
            "criteria": {
                "supported":    "the passage entails the hypothesis (entailment)",
                "unsupported":  "the passage is neutral — it neither confirms nor denies the hypothesis",
                "contradicted": "the passage contradicts the hypothesis",
            },
        },
    ]

    # Held-out benchmark wording — never seen during training
    BENCH_VARIANT = {
        "instructions": "Decide whether the cited text supports the claim.",
        "criteria": {
            "supported":    "the cited text confirms the claim",
            "unsupported":  "the cited text does not say enough to confirm or deny the claim",
            "contradicted": "the cited text states the opposite of the claim",
        },
    }

    # ------------------------------------------------------------------
    # Yesno variants (each QID gets a list of wording dicts)
    # ------------------------------------------------------------------
    YESNO_VARIANTS: dict[str, list[dict]] = {
        QID_MENTIONED: [
            {
                "instructions": (
                    "Does the evidence explicitly mention the topic of the claim?"
                ),
                "criteria": {
                    "yes": "the evidence explicitly addresses or names the topic of the claim",
                    "no":  "the evidence does not explicitly address the topic of the claim",
                },
            },
            {
                "instructions": (
                    "Is the subject of the claim directly referenced in the evidence?"
                ),
                "criteria": {
                    "yes": "the subject or key concept of the claim appears explicitly in the evidence",
                    "no":  "the subject or key concept is absent or only implied in the evidence",
                },
            },
        ],
        QID_NEEDS_MORE: [
            {
                "instructions": (
                    "Is additional evidence required to reach a confident verdict on the claim?"
                ),
                "criteria": {
                    "yes": "the available evidence is insufficient; more data or context is needed",
                    "no":  "the available evidence is sufficient to render a verdict",
                },
            },
            {
                "instructions": (
                    "Would a clinician need more information before acting on this claim?"
                ),
                "criteria": {
                    "yes": "yes, further test results, records, or context are required",
                    "no":  "no, the evidence on hand is sufficient to decide",
                },
            },
        ],
    }

    # ------------------------------------------------------------------
    # Score variants (single wording per QID — scores don't rotate)
    # ------------------------------------------------------------------
    SCORE_VARIANTS: dict[str, dict] = {
        QID_SUFFICIENCY: {
            "instructions": (
                "Rate how sufficient the evidence is for reaching a verdict on the claim. "
                "0.0 = completely insufficient, 1.0 = fully sufficient."
            ),
            "min_label": "completely insufficient",
            "max_label": "fully sufficient",
        },
        QID_SEVERITY: {
            "instructions": (
                "If the evidence contradicts the claim, rate the severity of the inconsistency. "
                "0.0 = no inconsistency, 1.0 = critical/dangerous inconsistency."
            ),
            "min_label": "no inconsistency",
            "max_label": "critical inconsistency",
        },
    }

    # ------------------------------------------------------------------
    # State builder
    # ------------------------------------------------------------------
    @staticmethod
    def state(claim: str, evidence: str) -> dict:
        """Build the laya state dict for a (claim, evidence) pair."""
        return {"claim": claim, "evidence": evidence}

    # ------------------------------------------------------------------
    # Question builders
    # ------------------------------------------------------------------
    @classmethod
    def question(cls, variant=None, option_order: Optional[list] = None) -> dict:
        """
        Build a laya-compatible choice question from a variant dict or index.

        Args:
            variant:      A dict from TRAIN_VARIANTS/BENCH_VARIANT, or int index into TRAIN_VARIANTS.
            option_order: Optional permutation of label indices for shuffled decoding.

        Returns:
            A dict ready to be passed to ``laya.train.to_internal``.
        """
        if variant is None:
            variant = cls.BENCH_VARIANT
        elif isinstance(variant, int):
            variant = cls.TRAIN_VARIANTS[variant]
        q: dict = {
            "type": "choice",
            "instructions": variant["instructions"],
            "criteria": {k: variant["criteria"][k] for k in cls.LABELS},
        }
        if option_order is not None:
            q["option_order"] = list(option_order)
        return q

    @classmethod
    def yesno_question(cls, qid: str, variant_idx: int = 0) -> dict:
        """
        Build a laya-compatible noul (boolean) question.

        Args:
            qid:         One of QID_MENTIONED / QID_NEEDS_MORE.
            variant_idx: Which wording to use (default 0).

        Returns:
            A dict with keys: type, instructions, criteria, labels.
        """
        variants = cls.YESNO_VARIANTS[qid]
        v = variants[variant_idx % len(variants)]
        return {
            "type": "noul",
            "instructions": v["instructions"],
            "criteria": {"true": v["criteria"]["yes"], "false": v["criteria"]["no"]},
            "labels": {"false": "no", "true": "yes"},
        }

    @classmethod
    def score_question(cls, qid: str) -> dict:
        """
        Build a laya-compatible score question.

        Args:
            qid: One of QID_SUFFICIENCY / QID_SEVERITY.

        Returns:
            A dict with keys: type, instructions, criteria.
        """
        v = cls.SCORE_VARIANTS[qid]
        return {
            "type": "score",
            "instructions": v["instructions"],
            "criteria": [v.get("min_label", "minimum"), v.get("max_label", "maximum")],
        }

    # ------------------------------------------------------------------
    # Convenience: label index ↔ name
    # ------------------------------------------------------------------
    @classmethod
    def label_index(cls, name: str) -> int:
        return list(cls.LABELS).index(name)

    @classmethod
    def label_name(cls, index: int) -> str:
        return cls.LABELS[index]
