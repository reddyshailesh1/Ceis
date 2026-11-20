"""
calibration.py — Calibration analysis for CEIS models.

Usage
-----
python evaluation/calibration.py \\
    --model runs/ceis_seed0 \\
    --split validation \\
    --config config/eval_config.yaml \\
    --out benchmark_out

Outputs
-------
benchmark_out/calibration_results.json  — ECE, MCE, Brier, temperature, bin data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from evaluation.metrics import (
    LABELS,
    brier_score,
    ece,
)

# ---------------------------------------------------------------------------
# Model loader (lazy; falls back to random probs when model not available)
# ---------------------------------------------------------------------------


def _load_model(model_name: str, config: dict):
    """
    Try to load the model via laya or transformers.
    Returns a callable  predict(pairs) -> np.ndarray of shape (N, 3).
    """
    try:
        from laya import Laya  # type: ignore

        laya = Laya(model_name)

        def predict(pairs: List[Tuple[str, str]]) -> np.ndarray:
            results = []
            for claim, evidence in pairs:
                res = laya.run({"claim": claim, "evidence": evidence})
                probs = _extract_probs(res)
                results.append(probs)
            return np.array(results)

        return predict
    except Exception:
        pass

    # Transformers / local checkpoint fallback
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        label_map = config.get("label_map", {})
        pipe = hf_pipeline("text-classification", model=model_name, top_k=None)

        def predict(pairs: List[Tuple[str, str]]) -> np.ndarray:  # type: ignore[misc]
            results = []
            for claim, evidence in pairs:
                text = f"{claim} [SEP] {evidence}"
                scores = pipe(text)[0]
                prob_vec = np.zeros(3)
                for item in scores:
                    raw_lbl = item["label"].lower()
                    mapped = label_map.get(raw_lbl, raw_lbl)
                    if mapped in LABELS:
                        prob_vec[LABELS.index(mapped)] = item["score"]
                # Normalise
                s = prob_vec.sum()
                if s > 0:
                    prob_vec /= s
                results.append(prob_vec)
            return np.array(results)

        return predict
    except Exception:
        pass

    # Stub — returns uniform distribution (for --help / dry-run testing)
    def predict(pairs: List[Tuple[str, str]]) -> np.ndarray:  # type: ignore[misc]
        n = len(pairs)
        p = np.full((n, 3), 1 / 3)
        return p

    return predict


def _extract_probs(result) -> np.ndarray:
    """Extract a 3-element probability vector from a laya result object."""
    prob_vec = np.zeros(3)
    try:
        # Try attribute access
        for i, label in enumerate(LABELS):
            prob_vec[i] = getattr(result, label, 0.0) or 0.0
    except Exception:
        pass
    s = prob_vec.sum()
    if s > 0:
        prob_vec /= s
    else:
        prob_vec = np.full(3, 1 / 3)
    return prob_vec


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def _load_split(split: str, config: dict) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return (pairs, labels) for the given split name."""
    import csv

    data_cfg = config.get("data", {})
    key = f"mednli_{split}"
    csv_path = Path(_ROOT) / data_cfg.get(key, f"data/splits/mednli_{split}.csv")

    if not csv_path.exists():
        # Return tiny synthetic stub so --help / imports still work
        pairs = [("Aspirin reduces fever.", "Patient was given aspirin for fever management.")]
        gold = ["supported"]
        return pairs, gold

    pairs, gold = [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pairs.append((row["claim"], row["evidence"]))
            gold.append(row["label"])
    return pairs, gold


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def _temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to logit-scaled probabilities."""
    # Convert probs → logits → scale → softmax
    eps = 1e-8
    logits = np.log(probs + eps)
    logits_scaled = logits / T
    # Softmax
    logits_shifted = logits_scaled - logits_scaled.max(axis=1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray,
    gold: Sequence[str],
    t_low: float = 0.1,
    t_high: float = 5.0,
    n_steps: int = 100,
) -> Tuple[float, float]:
    """
    Grid-search for optimal temperature T that minimises ECE on the calibration split.

    Returns (best_T, best_ece).
    """
    best_T = 1.0
    best_ece_val = float("inf")

    for T in np.linspace(t_low, t_high, n_steps):
        scaled = _temperature_scale(probs, float(T))
        ece_val, _, _ = ece(scaled, gold)
        if ece_val < best_ece_val:
            best_ece_val = ece_val
            best_T = float(T)

    return best_T, best_ece_val


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------


def run_calibration(
    model_name: str,
    split: str,
    config: dict,
    out_dir: Path,
) -> dict:
    """Run full calibration analysis and return results dict."""
    predict = _load_model(model_name, config)
    pairs, gold = _load_split(split, config)

    print(f"[calibration] Running inference on {len(pairs)} examples …")
    t0 = time.perf_counter()
    probs = predict(pairs)
    elapsed = time.perf_counter() - t0
    print(f"[calibration] Done in {elapsed:.2f}s")

    # --- Raw calibration ---
    ece_val, mce_val, bin_data = ece(probs, gold, n_bins=15)
    bs = brier_score(probs, gold)

    # --- Temperature scaling ---
    t_cfg = config.get("calibration", {})
    t_low = t_cfg.get("temperature_search_low", 0.1)
    t_high = t_cfg.get("temperature_search_high", 5.0)
    n_steps = t_cfg.get("temperature_search_steps", 100)

    best_T, post_ece = fit_temperature(probs, gold, t_low, t_high, n_steps)
    scaled_probs = _temperature_scale(probs, best_T)
    _, post_mce, post_bin_data = ece(scaled_probs, gold, n_bins=15)
    post_bs = brier_score(scaled_probs, gold)

    results = {
        "model": model_name,
        "split": split,
        "n_examples": len(pairs),
        "pre_temperature": {
            "ece": ece_val,
            "mce": mce_val,
            "brier_score": bs,
            "reliability_bins": bin_data,
        },
        "temperature_scaling": {
            "optimal_T": best_T,
            "post_ece": post_ece,
            "post_mce": post_mce,
            "post_brier_score": post_bs,
            "reliability_bins": post_bin_data,
        },
    }

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[calibration] Results saved to {out_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Model : {model_name}")
    print(f"Split : {split}  (n={len(pairs)})")
    print(f"{'─'*60}")
    print(f"Pre-scaling  ECE={ece_val:.4f}  MCE={mce_val:.4f}  Brier={bs:.4f}")
    print(f"Temperature  T={best_T:.4f}")
    print(f"Post-scaling ECE={post_ece:.4f}  MCE={post_mce:.4f}  Brier={post_bs:.4f}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibration.py",
        description="Calibration analysis for CEIS models (ECE, MCE, Brier, temperature scaling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model name or local checkpoint path (e.g. runs/ceis_seed0).",
    )
    p.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on.",
    )
    p.add_argument(
        "--config",
        default="config/eval_config.yaml",
        help="Path to eval_config.yaml.",
    )
    p.add_argument(
        "--out",
        default="benchmark_out",
        help="Output directory for calibration_results.json.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(_ROOT) / args.config
    config: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

    out_dir = Path(_ROOT) / args.out

    run_calibration(
        model_name=args.model,
        split=args.split,
        config=config,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Compatibility class for main.py
# ---------------------------------------------------------------------------
class CalibrationAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    def run(self):
        m = self.config.get("models", {}).get("ceis", {}).get("name", "runs/ceis_seed0")
        out_dir = Path(_ROOT) / self.config.get("output", {}).get("calibration_dir", "benchmark_out")
        return run_calibration(model_name=m, split="validation", config=self.config, out_dir=out_dir)
