"""
clean.py — Cleaning pipeline for CEIS datasets.

Steps:
  MedNLI:
    1. Load parquet files (train / validation / test)
    2. Map gold_label: entailment→supported, neutral→unsupported,
       contradiction→contradicted
    3. Remove conflicting labels (same (claim, evidence) pair with >1 label)
    4. Deduplicate on (claim, evidence) — keep first occurrence
    5. De-id marker handling: keep [** ... **] patterns as-is (default) or mask

  PubMedQA:
    1. Load parquet
    2. Map final_decision: yes→supported, no→contradicted, maybe→unsupported
    3. Keep long_answer as evidence, question as claim
    4. Same conflict/dedup logic

Outputs:
  data/processed/mednli_train.csv
  data/processed/mednli_validation.csv
  data/processed/mednli_test.csv
  data/processed/pubmedqa_train.csv

Usage:
    python data/clean.py --raw data/raw --out data/processed
    python data/clean.py --raw data/raw --out data/processed --mask-deids
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEDNLI_LABEL_MAP = {
    "entailment": "supported",
    "neutral": "unsupported",
    "contradiction": "contradicted",
}

PUBMEDQA_LABEL_MAP = {
    "yes": "supported",
    "no": "contradicted",
    "maybe": "unsupported",
}

DEID_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")
DEID_REPLACEMENT = "[DEID]"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def mask_deids(text: str) -> str:
    """Replace [** ... **] de-identification markers with [DEID]."""
    return DEID_PATTERN.sub(DEID_REPLACEMENT, text)


def clean_text(text: str, mask: bool = False) -> str:
    """Normalize whitespace and optionally mask de-id markers."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if mask:
        text = mask_deids(text)
    return text


def remove_conflicts_and_dedup(
    df: pd.DataFrame,
    claim_col: str = "claim",
    evidence_col: str = "evidence",
    label_col: str = "label",
) -> tuple[pd.DataFrame, int, int]:
    """
    Remove rows where the same (claim, evidence) pair has multiple labels.
    Then deduplicate, keeping first occurrence.

    Returns:
        (cleaned_df, n_conflicting_dropped, n_dupes_dropped)
    """
    rows_before = len(df)

    # Identify conflicting pairs
    pair_label_counts = (
        df.groupby([claim_col, evidence_col])[label_col]
        .nunique()
        .reset_index(name="n_labels")
    )
    conflicting_pairs = pair_label_counts[pair_label_counts["n_labels"] > 1][
        [claim_col, evidence_col]
    ]
    conflict_key = set(
        zip(conflicting_pairs[claim_col], conflicting_pairs[evidence_col])
    )

    df_no_conflicts = df[
        ~df.apply(
            lambda r: (r[claim_col], r[evidence_col]) in conflict_key, axis=1
        )
    ].copy()
    n_conflicting_dropped = rows_before - len(df_no_conflicts)

    # Deduplicate
    df_deduped = df_no_conflicts.drop_duplicates(
        subset=[claim_col, evidence_col], keep="first"
    )
    n_dupes_dropped = len(df_no_conflicts) - len(df_deduped)

    return df_deduped.reset_index(drop=True), n_conflicting_dropped, n_dupes_dropped


def print_stats(
    dataset: str,
    split: str,
    rows_in: int,
    conflicting_dropped: int,
    dupes_dropped: int,
    rows_out: int,
) -> None:
    logger.info(
        "[%s/%s] rows_in=%d  conflicting_dropped=%d  dupes_dropped=%d  rows_out=%d",
        dataset,
        split,
        rows_in,
        conflicting_dropped,
        dupes_dropped,
        rows_out,
    )


# ---------------------------------------------------------------------------
# MedNLI cleaning
# ---------------------------------------------------------------------------


def clean_mednli_split(
    parquet_path: Path,
    split: str,
    out_dir: Path,
    mask: bool = False,
) -> Optional[pd.DataFrame]:
    """Load, clean, and save one MedNLI split. Returns the cleaned DataFrame."""
    if not parquet_path.exists():
        logger.warning("MedNLI %s parquet not found: %s — skipping", split, parquet_path)
        return None

    df = pd.read_parquet(parquet_path)
    rows_in = len(df)

    # Identify claim / evidence / label columns (column names vary by upload)
    # Typical MedNLI columns: sentence1, sentence2, gold_label
    # Also seen: premise, hypothesis, label
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    for candidate in ["sentence1", "premise"]:
        if candidate in lower_cols:
            col_map["evidence"] = lower_cols[candidate]
            break
    for candidate in ["sentence2", "hypothesis"]:
        if candidate in lower_cols:
            col_map["claim"] = lower_cols[candidate]
            break
    for candidate in ["gold_label", "label"]:
        if candidate in lower_cols:
            col_map["label_raw"] = lower_cols[candidate]
            break

    if len(col_map) < 3:
        logger.error(
            "Cannot detect required columns in %s. Found: %s", parquet_path, list(df.columns)
        )
        return None

    df = df.rename(
        columns={
            col_map["evidence"]: "evidence",
            col_map["claim"]: "claim",
            col_map["label_raw"]: "label_raw",
        }
    )

    # Clean text
    df["claim"] = df["claim"].apply(lambda t: clean_text(t, mask=mask))
    df["evidence"] = df["evidence"].apply(lambda t: clean_text(t, mask=mask))

    # Map labels
    df["label"] = df["label_raw"].str.lower().str.strip().map(MEDNLI_LABEL_MAP)
    unknown_labels = df[df["label"].isna()]["label_raw"].unique().tolist()
    if unknown_labels:
        logger.warning("Unknown MedNLI labels (dropped): %s", unknown_labels)
    df = df[df["label"].notna()].copy()

    # Keep only needed columns + any extra we want
    df = df[["claim", "evidence", "label"]].copy()

    # Remove conflicts and deduplicate
    df, n_conflicts, n_dupes = remove_conflicts_and_dedup(df)
    rows_out = len(df)

    print_stats("mednli", split, rows_in, n_conflicts, n_dupes, rows_out)

    # Save
    out_path = out_dir / f"mednli_{split}.csv"
    df.to_csv(out_path, index=False)
    logger.info("  → saved %s", out_path)

    return df


def clean_mednli(raw_dir: Path, out_dir: Path, mask: bool = False) -> None:
    """Clean all three MedNLI splits."""
    mednli_dir = raw_dir / "mednli"
    for split in ["train", "validation", "test"]:
        parquet_path = mednli_dir / f"{split}.parquet"
        clean_mednli_split(parquet_path, split, out_dir, mask=mask)


# ---------------------------------------------------------------------------
# PubMedQA cleaning
# ---------------------------------------------------------------------------


def clean_pubmedqa(raw_dir: Path, out_dir: Path, mask: bool = False) -> None:
    """Load, clean, and save PubMedQA."""
    parquet_path = raw_dir / "pubmedqa" / "train.parquet"
    if not parquet_path.exists():
        logger.warning("PubMedQA parquet not found: %s — skipping", parquet_path)
        return

    df = pd.read_parquet(parquet_path)
    rows_in = len(df)

    # PubMedQA columns: pubid, question, context, long_answer, final_decision
    # Flatten context if it's nested (list of strings or dict)
    if "long_answer" not in df.columns:
        logger.error("long_answer column not found in PubMedQA. Columns: %s", list(df.columns))
        return

    if "question" not in df.columns:
        logger.error("question column not found in PubMedQA. Columns: %s", list(df.columns))
        return

    if "final_decision" not in df.columns:
        # Try alternative column names
        for alt in ["answer", "label", "decision"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "final_decision"})
                break
        else:
            logger.error("Cannot find decision column in PubMedQA. Columns: %s", list(df.columns))
            return

    # Extract pubid for group-aware splitting later
    if "pubid" not in df.columns:
        df["pubid"] = df.index.astype(str)

    # Build claim and evidence
    df["claim"] = df["question"].apply(lambda t: clean_text(t, mask=mask))

    def extract_evidence(row):
        la = row.get("long_answer", "")
        if isinstance(la, str):
            return clean_text(la, mask=mask)
        # If long_answer is a list, join
        if isinstance(la, (list, tuple)):
            return clean_text(" ".join(str(x) for x in la), mask=mask)
        return ""

    df["evidence"] = df.apply(extract_evidence, axis=1)

    # Map labels
    df["label"] = df["final_decision"].str.lower().str.strip().map(PUBMEDQA_LABEL_MAP)
    unknown_labels = df[df["label"].isna()]["final_decision"].unique().tolist()
    if unknown_labels:
        logger.warning("Unknown PubMedQA labels (dropped): %s", unknown_labels)
    df = df[df["label"].notna()].copy()

    # Keep pubid for splitting
    df = df[["pubid", "claim", "evidence", "label"]].copy()

    # Remove conflicts and dedup
    df_clean, n_conflicts, n_dupes = remove_conflicts_and_dedup(df)
    rows_out = len(df_clean)

    print_stats("pubmedqa", "train", rows_in, n_conflicts, n_dupes, rows_out)

    out_path = out_dir / "pubmedqa_train.csv"
    df_clean.to_csv(out_path, index=False)
    logger.info("  → saved %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw CEIS datasets (MedNLI & PubMedQA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        default="data/raw",
        help="Root directory containing raw data (mednli/, pubmedqa/ subdirs).",
    )
    parser.add_argument(
        "--out",
        default="data/processed",
        help="Output directory for cleaned CSV files.",
    )
    parser.add_argument(
        "--mask-deids",
        action="store_true",
        help="Replace [** ... **] de-identification markers with [DEID].",
    )
    parser.add_argument(
        "--skip-mednli",
        action="store_true",
        help="Skip MedNLI cleaning.",
    )
    parser.add_argument(
        "--skip-pubmedqa",
        action="store_true",
        help="Skip PubMedQA cleaning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Raw directory : %s", raw_dir.resolve())
    logger.info("Output directory: %s", out_dir.resolve())
    logger.info("Mask de-ids : %s", args.mask_deids)

    if not args.skip_mednli:
        clean_mednli(raw_dir, out_dir, mask=args.mask_deids)
    else:
        logger.info("Skipping MedNLI cleaning.")

    if not args.skip_pubmedqa:
        clean_pubmedqa(raw_dir, out_dir, mask=args.mask_deids)
    else:
        logger.info("Skipping PubMedQA cleaning.")

    logger.info("Cleaning complete.")


if __name__ == "__main__":
    main()
