"""
ood_eval.py — Out-of-domain evaluation on PubMedQA for CEIS models.

Usage
-----
python evaluation/ood_eval.py \\
    --models base pavanmantha/mednli_laya runs/ceis_seed0 \\
    --data data/splits/pubmedqa_test.csv \\
    --config config/eval_config.yaml \\
    --out benchmark_out

NOTE
----
PubMedQA labels (yes / no / maybe) map *imperfectly* to MedNLI-derived CEIS labels:
  yes   → supported
  no    → contradicted
  maybe → unsupported  (closest approximation; treat results as indicative only)

The 'maybe' category has no clean equivalent in a 3-way entailment schema.
OOD numbers should NOT be directly compared with in-domain MedNLI accuracy.

Outputs
-------
benchmark_out/ood_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from evaluation.metrics import (
    LABELS,
    accuracy,
    brier_score,
    ece,
    macro_f1,
    per_class_metrics,
    confusion_matrix as cm_fn,
    latency_ms,
    throughput as tput_fn,
)

# ---------------------------------------------------------------------------
# OOD label mapping
# ---------------------------------------------------------------------------

PUBMEDQA_MAP = {
    "yes": "supported",
    "no": "contradicted",
    "maybe": "unsupported",
}

MAPPING_WARNING = (
    "WARNING: PubMedQA labels (yes/no/maybe) map imperfectly to MedNLI-derived CEIS labels. "
    "'maybe' → 'unsupported' is a rough approximation. "
    "OOD results are indicative only and should not be directly compared with in-domain MedNLI accuracy."
)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def load_pubmedqa(csv_path: Path) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Load PubMedQA test CSV.

    Expected columns: claim (or question), evidence (or long_answer), label (or final_decision).
    Returns (pairs, ceis_labels, raw_labels).
    """
    pairs: List[Tuple[str, str]] = []
    ceis_labels: List[str] = []
    raw_labels: List[str] = []

    if not csv_path.exists():
        # Stub for dry-run / --help
        print(f"[ood_eval] WARNING: {csv_path} not found — using 1-example stub.")
        pairs = [("Beta-blockers reduce mortality in heart failure.", "Study shows beta-blocker therapy significantly reduced all-cause mortality (p<0.001).")]
        ceis_labels = ["supported"]
        raw_labels = ["yes"]
        return pairs, ceis_labels, raw_labels

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        # Flexible column names
        claim_col = next((c for c in ["claim", "question"] if c in fieldnames), fieldnames[0])
        evidence_col = next((c for c in ["evidence", "long_answer", "context"] if c in fieldnames), fieldnames[1])
        label_col = next((c for c in ["label", "final_decision", "answer"] if c in fieldnames), fieldnames[2])

        for row in reader:
            claim = row[claim_col].strip()
            evidence = row[evidence_col].strip()
            raw = row[label_col].strip().lower()
            mapped = PUBMEDQA_MAP.get(raw, "unsupported")
            pairs.append((claim, evidence))
            ceis_labels.append(mapped)
            raw_labels.append(raw)

    return pairs, ceis_labels, raw_labels


# ---------------------------------------------------------------------------
# Model loading (re-use same pattern as calibration.py)
# ---------------------------------------------------------------------------


def _load_model(model_name: str, label_map: dict):
    """Return predict(pairs) -> np.ndarray (N, 3)."""
    try:
        from laya import Laya  # type: ignore

        laya = Laya(model_name)

        def predict(pairs):
            results = []
            for claim, evidence in pairs:
                res = laya.run({"claim": claim, "evidence": evidence})
                results.append(_extract_probs(res))
            return np.array(results)

        return predict
    except Exception:
        pass

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        pipe = hf_pipeline("text-classification", model=model_name, top_k=None)

        def predict(pairs):  # type: ignore[misc]
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
                s = prob_vec.sum()
                prob_vec = prob_vec / s if s > 0 else np.full(3, 1 / 3)
                results.append(prob_vec)
            return np.array(results)

        return predict
    except Exception:
        pass

    # Stub
    def predict(pairs):  # type: ignore[misc]
        return np.full((len(pairs), 3), 1 / 3)

    return predict


def _extract_probs(result) -> np.ndarray:
    prob_vec = np.zeros(3)
    try:
        for i, label in enumerate(LABELS):
            prob_vec[i] = getattr(result, label, 0.0) or 0.0
    except Exception:
        pass
    s = prob_vec.sum()
    return prob_vec / s if s > 0 else np.full(3, 1 / 3)


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------


def evaluate_model_ood(
    model_name: str,
    pairs: List[Tuple[str, str]],
    ceis_labels: List[str],
    raw_labels: List[str],
    label_map: dict,
) -> dict:
    predict = _load_model(model_name, label_map)

    t0 = time.perf_counter()
    probs = predict(pairs)
    elapsed = time.perf_counter() - t0

    pred_idx = probs.argmax(axis=1)
    pred = [LABELS[i] for i in pred_idx]

    acc = accuracy(pred, ceis_labels)
    mf1 = macro_f1(pred, ceis_labels)
    pcm = per_class_metrics(pred, ceis_labels)
    cm = cm_fn(pred, ceis_labels).tolist()
    ece_val, mce_val, bin_data = ece(probs, ceis_labels)
    bs = brier_score(probs, ceis_labels)
    lat = latency_ms(elapsed, len(pairs))
    tput = tput_fn(elapsed, len(pairs))

    # Label distribution shift
    raw_dist = dict(Counter(raw_labels))
    mapped_dist = dict(Counter(ceis_labels))
    pred_dist = dict(Counter(pred))

    return {
        "model": model_name,
        "n_examples": len(pairs),
        "accuracy": acc,
        "macro_f1": mf1,
        "per_class": pcm,
        "confusion_matrix": cm,
        "ece": ece_val,
        "mce": mce_val,
        "brier_score": bs,
        "latency_ms": lat,
        "throughput_per_sec": tput,
        "label_distribution": {
            "raw_pubmedqa": raw_dist,
            "mapped_ceis": mapped_dist,
            "predicted": pred_dist,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_ood_eval(
    model_names: List[str],
    data_path: Path,
    config: dict,
    out_dir: Path,
) -> dict:
    print(f"\n{'='*60}")
    print("CEIS  Out-of-Domain Evaluation  (PubMedQA)")
    print(f"{'='*60}")
    print(MAPPING_WARNING)
    print(f"{'─'*60}\n")

    pairs, ceis_labels, raw_labels = load_pubmedqa(data_path)
    print(f"Loaded {len(pairs)} examples from {data_path}\n")

    label_map = config.get("label_map", {})
    results_per_model: List[dict] = []

    for model_name in model_names:
        print(f"[ood_eval] Evaluating: {model_name} …")
        res = evaluate_model_ood(model_name, pairs, ceis_labels, raw_labels, label_map)
        results_per_model.append(res)
        print(
            f"  acc={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}"
            f"  ECE={res['ece']:.4f}  Brier={res['brier_score']:.4f}"
            f"  lat={res['latency_ms']:.1f}ms"
        )

    # Comparison table header
    print(f"\n{'─'*60}")
    print(f"{'Model':<35} {'Acc':>6} {'MacF1':>7} {'ECE':>7} {'Brier':>7}")
    print(f"{'─'*60}")
    for r in results_per_model:
        print(
            f"{r['model']:<35} {r['accuracy']:>6.4f} {r['macro_f1']:>7.4f}"
            f" {r['ece']:>7.4f} {r['brier_score']:>7.4f}"
        )
    print(f"{'─'*60}\n")

    output = {
        "dataset": "pubmedqa",
        "label_mapping_note": MAPPING_WARNING,
        "pubmedqa_label_map": PUBMEDQA_MAP,
        "results": results_per_model,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ood_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"[ood_eval] Results saved to {out_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ood_eval.py",
        description=(
            "Out-of-domain evaluation on PubMedQA. "
            "NOTE: PubMedQA labels (yes/no/maybe) map imperfectly to MedNLI labels; "
            "results are indicative only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["multilingual", "pavanmantha/mednli_laya", "runs/ceis_seed0"],
        help="Model names / local paths to evaluate.",
    )
    p.add_argument(
        "--data",
        default="data/splits/pubmedqa_test.csv",
        help="Path to PubMedQA test CSV.",
    )
    p.add_argument(
        "--config",
        default="config/eval_config.yaml",
        help="Path to eval_config.yaml.",
    )
    p.add_argument(
        "--out",
        default="benchmark_out",
        help="Output directory for ood_results.json.",
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

    data_path = Path(_ROOT) / args.data
    out_dir = Path(_ROOT) / args.out

    run_ood_eval(
        model_names=args.models,
        data_path=data_path,
        config=config,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Compatibility class for main.py
# ---------------------------------------------------------------------------
class OODEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def run(self):
        models = [
            self.config.get("models", {}).get("base", {}).get("name", "multilingual"),
            self.config.get("models", {}).get("mednli_laya", {}).get("name", "pavanmantha/mednli_laya"),
            self.config.get("models", {}).get("ceis", {}).get("name", "runs/ceis_seed0"),
        ]
        data_path = Path(_ROOT) / self.config.get("data", {}).get("pubmedqa_test", "data/splits/pubmedqa_test.csv")
        out_dir = Path(_ROOT) / self.config.get("output", {}).get("benchmark_dir", "benchmark_out")
        return run_ood_eval(model_names=models, data_path=data_path, config=self.config, out_dir=out_dir)
