"""
download.py — Download raw datasets for CEIS.

Downloads:
  - MedNLI  (presencesw/mednli)      → data/raw/mednli/
  - PubMedQA (qiaojin/PubMedQA, pqa_labeled) → data/raw/pubmedqa/

Usage:
    python data/download.py --out data/raw
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MedNLI helpers
# ---------------------------------------------------------------------------

MEDNLI_REPO = "presencesw/mednli"
MEDNLI_SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def download_mednli(out_dir: Path) -> None:
    """Download all three MedNLI splits from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )

    mednli_dir = out_dir / "mednli"
    mednli_dir.mkdir(parents=True, exist_ok=True)

    for split_name, repo_path in MEDNLI_SPLITS.items():
        dest = mednli_dir / f"{split_name}.parquet"
        if dest.exists():
            logger.info("MedNLI %s already exists at %s — skipping", split_name, dest)
            continue

        logger.info("Downloading MedNLI %s …", split_name)
        try:
            cached = hf_hub_download(
                repo_id=MEDNLI_REPO,
                filename=repo_path,
                repo_type="dataset",
            )
            shutil.copy2(cached, dest)
            logger.info("  → saved %s (%d bytes)", dest, dest.stat().st_size)
        except Exception as exc:
            logger.warning("  Failed to download MedNLI %s: %s", split_name, exc)
            # Try alternative path patterns
            alt_paths = [
                f"{split_name}-00000-of-00001.parquet",
                f"{split_name}.parquet",
            ]
            downloaded = False
            for alt in alt_paths:
                try:
                    cached = hf_hub_download(
                        repo_id=MEDNLI_REPO,
                        filename=alt,
                        repo_type="dataset",
                    )
                    shutil.copy2(cached, dest)
                    logger.info("  → saved %s via alt path (%d bytes)", dest, dest.stat().st_size)
                    downloaded = True
                    break
                except Exception:
                    continue
            if not downloaded:
                logger.error("  Could not download MedNLI %s — skipping", split_name)


# ---------------------------------------------------------------------------
# PubMedQA helpers
# ---------------------------------------------------------------------------

PUBMEDQA_REPO = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
# Typical HF datasets-server parquet path for pqa_labeled
PUBMEDQA_PARQUET_PATHS = [
    f"{PUBMEDQA_CONFIG}/train/train-00000-of-00001.parquet",
    f"{PUBMEDQA_CONFIG}/train-00000-of-00001.parquet",
    "train-00000-of-00001.parquet",
]


def download_pubmedqa(out_dir: Path) -> None:
    """Download PubMedQA (pqa_labeled) from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )

    pubmedqa_dir = out_dir / "pubmedqa"
    pubmedqa_dir.mkdir(parents=True, exist_ok=True)

    dest = pubmedqa_dir / "train.parquet"
    if dest.exists():
        logger.info("PubMedQA already exists at %s — skipping", dest)
        return

    logger.info("Downloading PubMedQA (pqa_labeled) …")

    # Try known parquet paths first
    downloaded = False
    for path in PUBMEDQA_PARQUET_PATHS:
        try:
            cached = hf_hub_download(
                repo_id=PUBMEDQA_REPO,
                filename=path,
                repo_type="dataset",
            )
            shutil.copy2(cached, dest)
            logger.info("  → saved %s (%d bytes)", dest, dest.stat().st_size)
            downloaded = True
            break
        except Exception as exc:
            logger.debug("  Path %s failed: %s", path, exc)

    if not downloaded:
        # Fall back: load with datasets library and export to parquet
        logger.info("  Falling back to datasets library …")
        try:
            import datasets as hf_datasets

            ds = hf_datasets.load_dataset(
                PUBMEDQA_REPO,
                PUBMEDQA_CONFIG,
                split="train",
                trust_remote_code=True,
            )
            df = ds.to_pandas()
            df.to_parquet(dest, index=False)
            logger.info(
                "  → saved %s via datasets (%d rows, %d bytes)",
                dest,
                len(df),
                dest.stat().st_size,
            )
            downloaded = True
        except Exception as exc2:
            logger.error("  datasets fallback also failed: %s", exc2)

    if not downloaded:
        logger.error(
            "Could not download PubMedQA. "
            "Install datasets: pip install datasets  and retry."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw CEIS datasets (MedNLI & PubMedQA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Root output directory for raw data.",
    )
    parser.add_argument(
        "--skip-mednli",
        action="store_true",
        help="Skip MedNLI download.",
    )
    parser.add_argument(
        "--skip-pubmedqa",
        action="store_true",
        help="Skip PubMedQA download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", out_dir.resolve())

    if not args.skip_mednli:
        download_mednli(out_dir)
    else:
        logger.info("Skipping MedNLI download.")

    if not args.skip_pubmedqa:
        download_pubmedqa(out_dir)
    else:
        logger.info("Skipping PubMedQA download.")

    logger.info("Done. Files written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
