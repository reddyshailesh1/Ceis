"""
split.py — Group-aware train/val/test splitting for CEIS datasets.

MedNLI:
  - Split is pre-defined (train / validation / test from clean.py).
  - Verify no premise (evidence) leakage across splits.

PubMedQA:
  - 80/10/10 split by pubid (group) to prevent evidence leakage.

Output columns per CSV: claim, evidence, label, group, source

Usage:
    python data/split.py --processed data/processed --out data/splits
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_COLS = {"claim", "evidence", "label"}
OUTPUT_COLS = ["claim", "evidence", "label", "group", "source"]


# ---------------------------------------------------------------------------
# MedNLI splitting
# ---------------------------------------------------------------------------


def process_mednli(processed_dir: Path, out_dir: Path) -> None:
    """
    Load pre-split MedNLI CSVs, verify no premise leakage, and save with
    standard schema (claim, evidence, label, group, source).
    """
    split_dfs: dict[str, Optional[pd.DataFrame]] = {}

    for split in ["train", "validation", "test"]:
        csv_path = processed_dir / f"mednli_{split}.csv"
        if not csv_path.exists():
            logger.warning("MedNLI %s CSV not found: %s — skipping", split, csv_path)
            split_dfs[split] = None
            continue
        df = pd.read_csv(csv_path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            logger.error("MedNLI %s missing columns: %s", split, missing)
            split_dfs[split] = None
            continue
        df["group"] = df["evidence"]  # group = premise (evidence)
        df["source"] = "mednli"
        split_dfs[split] = df

    # --- Leakage check ---
    present_splits = {k: v for k, v in split_dfs.items() if v is not None}
    split_names = list(present_splits.keys())
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1 :]:
            premises1 = set(present_splits[s1]["evidence"])
            premises2 = set(present_splits[s2]["evidence"])
            overlap = premises1 & premises2
            if overlap:
                logger.warning(
                    "Premise leakage between MedNLI %s and %s: %d shared premises",
                    s1,
                    s2,
                    len(overlap),
                )
            else:
                logger.info(
                    "✓ No premise leakage between MedNLI %s and %s", s1, s2
                )

    # --- Save ---
    for split, df in present_splits.items():
        out_path = out_dir / f"mednli_{split}.csv"
        df[OUTPUT_COLS].to_csv(out_path, index=False)
        label_dist = df["label"].value_counts().to_dict()
        logger.info(
            "[mednli/%s] rows=%d  labels=%s  saved→%s",
            split,
            len(df),
            label_dist,
            out_path,
        )


# ---------------------------------------------------------------------------
# PubMedQA splitting
# ---------------------------------------------------------------------------


def process_pubmedqa(
    processed_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Load cleaned PubMedQA, do group-aware 80/10/10 split by pubid, and save.
    """
    csv_path = processed_dir / "pubmedqa_train.csv"
    if not csv_path.exists():
        logger.warning("PubMedQA CSV not found: %s — skipping", csv_path)
        return

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error("PubMedQA CSV missing columns: %s", missing)
        return

    # pubid column expected from clean.py; fallback to index
    if "pubid" not in df.columns:
        logger.warning("pubid column missing; using row index as group ID")
        df["pubid"] = df.index.astype(str)

    df["group"] = df["pubid"].astype(str)
    df["source"] = "pubmedqa"

    # Group-aware split: shuffle unique pubids, then slice
    rng = np.random.default_rng(seed)
    unique_groups = df["pubid"].unique()
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(np.floor(train_ratio * n))
    n_val = int(np.floor(val_ratio * n))
    # remainder goes to test

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    df_train = df[df["pubid"].isin(train_groups)].copy()
    df_val = df[df["pubid"].isin(val_groups)].copy()
    df_test = df[df["pubid"].isin(test_groups)].copy()

    # Leakage check
    for s1_name, s1_grps in [("train", train_groups), ("val", val_groups), ("test", test_groups)]:
        for s2_name, s2_grps in [("train", train_groups), ("val", val_groups), ("test", test_groups)]:
            if s1_name >= s2_name:
                continue
            overlap = s1_grps & s2_grps
            if overlap:
                logger.error(
                    "Group leakage between PubMedQA %s and %s: %d groups!",
                    s1_name,
                    s2_name,
                    len(overlap),
                )
            else:
                logger.info("✓ No group leakage between PubMedQA %s and %s", s1_name, s2_name)

    # Save
    for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out_path = out_dir / f"pubmedqa_{split_name}.csv"
        df_split[OUTPUT_COLS].to_csv(out_path, index=False)
        label_dist = df_split["label"].value_counts().to_dict()
        logger.info(
            "[pubmedqa/%s] groups=%d  rows=%d  labels=%s  saved→%s",
            split_name,
            df_split["pubid"].nunique(),
            len(df_split),
            label_dist,
            out_path,
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(out_dir: Path) -> None:
    """Print a summary table of all split files."""
    logger.info("\n--- Split Summary ---")
    total = 0
    for csv_path in sorted(out_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            label_dist = df["label"].value_counts().to_dict()
            logger.info(
                "  %-40s  rows=%-6d  %s",
                csv_path.name,
                len(df),
                label_dist,
            )
            total += len(df)
        except Exception as exc:
            logger.warning("  Could not read %s: %s", csv_path.name, exc)
    logger.info("  TOTAL rows: %d", total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group-aware splitting of cleaned CEIS datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed",
        default="data/processed",
        help="Directory containing cleaned CSV files from clean.py.",
    )
    parser.add_argument(
        "--out",
        default="data/splits",
        help="Output directory for split CSV files.",
    )
    parser.add_argument(
        "--pubmedqa-train-ratio",
        type=float,
        default=0.8,
        help="Fraction of PubMedQA groups to assign to train.",
    )
    parser.add_argument(
        "--pubmedqa-val-ratio",
        type=float,
        default=0.1,
        help="Fraction of PubMedQA groups to assign to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PubMedQA group shuffling.",
    )
    parser.add_argument(
        "--skip-mednli",
        action="store_true",
        help="Skip MedNLI processing.",
    )
    parser.add_argument(
        "--skip-pubmedqa",
        action="store_true",
        help="Skip PubMedQA processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processed directory: %s", processed_dir.resolve())
    logger.info("Output directory   : %s", out_dir.resolve())

    if not args.skip_mednli:
        process_mednli(processed_dir, out_dir)
    else:
        logger.info("Skipping MedNLI processing.")

    if not args.skip_pubmedqa:
        process_pubmedqa(
            processed_dir,
            out_dir,
            train_ratio=args.pubmedqa_train_ratio,
            val_ratio=args.pubmedqa_val_ratio,
            seed=args.seed,
        )
    else:
        logger.info("Skipping PubMedQA processing.")

    print_summary(out_dir)
    logger.info("Splitting complete.")


if __name__ == "__main__":
    main()
