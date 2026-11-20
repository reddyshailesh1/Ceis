"""
Shared evaluation metrics for the Clinical Evidence Intelligence System (CEIS).

Provides pure-Python / NumPy implementations of all metrics needed by
benchmark.py, calibration.py, and ood_eval.py.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABELS = ["supported", "unsupported", "contradicted"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Basic classification metrics
# ---------------------------------------------------------------------------


def accuracy(pred: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of predictions that match gold labels."""
    if len(pred) != len(gold):
        raise ValueError("pred and gold must have the same length")
    if not pred:
        return 0.0
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def _tp_fp_fn(
    pred: Sequence[str],
    gold: Sequence[str],
    label: str,
) -> Tuple[int, int, int]:
    tp = sum(p == g == label for p, g in zip(pred, gold))
    fp = sum(p == label and g != label for p, g in zip(pred, gold))
    fn = sum(g == label and p != label for p, g in zip(pred, gold))
    return tp, fp, fn


def precision_recall_f1(
    pred: Sequence[str],
    gold: Sequence[str],
    label: str,
) -> Tuple[float, float, float]:
    """Precision, recall, F1 for a single label."""
    tp, fp, fn = _tp_fp_fn(pred, gold, label)
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def macro_f1(
    pred: Sequence[str],
    gold: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> float:
    """Macro-averaged F1 across all labels."""
    if labels is None:
        labels = LABELS
    f1s = [precision_recall_f1(pred, gold, l)[2] for l in labels]
    return float(np.mean(f1s))


def per_class_metrics(
    pred: Sequence[str],
    gold: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return dict of {label: {precision, recall, f1, support}} for each label."""
    if labels is None:
        labels = LABELS
    gold_counts = Counter(gold)
    result: Dict[str, Dict[str, float]] = {}
    for label in labels:
        p, r, f = precision_recall_f1(pred, gold, label)
        result[label] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": gold_counts.get(label, 0),
        }
    return result


def confusion_matrix(
    pred: Sequence[str],
    gold: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Return a K×K confusion matrix as a NumPy array.

    Row = gold label, column = predicted label.
    Labels order follows `labels` argument (defaults to LABELS).
    """
    if labels is None:
        labels = LABELS
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    mat = np.zeros((k, k), dtype=int)
    for p, g in zip(pred, gold):
        if g in idx and p in idx:
            mat[idx[g], idx[p]] += 1
    return mat


# ---------------------------------------------------------------------------
# Probabilistic calibration metrics
# ---------------------------------------------------------------------------


def brier_score(
    probs: np.ndarray,
    gold: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> float:
    """
    Multi-class Brier score (mean squared error of probability vectors).

    Parameters
    ----------
    probs : ndarray of shape (N, K)
        Predicted probability distributions over labels.
    gold  : sequence of str
        True labels (length N).
    labels : list of str, optional
        Label order matching probs columns (defaults to LABELS).
    """
    if labels is None:
        labels = LABELS
    idx = {l: i for i, l in enumerate(labels)}
    n, k = probs.shape
    one_hot = np.zeros((n, k))
    for i, g in enumerate(gold):
        if g in idx:
            one_hot[i, idx[g]] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ece(
    probs: np.ndarray,
    gold: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    n_bins: int = 15,
) -> Tuple[float, float, List[Dict]]:
    """
    Expected Calibration Error (equal-width bins on max-class confidence).

    Returns
    -------
    ece_val : float
    mce_val : float   – Maximum Calibration Error
    bin_data : list of dicts  – reliability diagram data per bin
    """
    if labels is None:
        labels = LABELS
    idx = {l: i for i, l in enumerate(labels)}
    n = len(gold)

    max_conf = probs.max(axis=1)
    pred_class = probs.argmax(axis=1)
    correct = np.array([pred_class[i] == idx.get(g, -1) for i, g in enumerate(gold)], dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    weighted_error = 0.0
    max_cal_err = 0.0

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (max_conf >= lo) & (max_conf < hi if b < n_bins - 1 else max_conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_data.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0})
            continue
        acc = float(correct[mask].mean())
        conf = float(max_conf[mask].mean())
        gap = abs(acc - conf)
        weighted_error += (count / n) * gap
        max_cal_err = max(max_cal_err, gap)
        bin_data.append({"bin_lo": lo, "bin_hi": hi, "count": count, "accuracy": acc, "confidence": conf, "gap": gap})

    return float(weighted_error), float(max_cal_err), bin_data


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    correct: Sequence[int],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI on accuracy.

    Parameters
    ----------
    correct : sequence of 0/1 integers
    n_boot  : number of resamples
    seed    : RNG seed

    Returns
    -------
    point_estimate, ci_low, ci_high
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct, dtype=float)
    n = len(arr)
    point = arr.mean()
    boot_means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def mcnemar_test(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
) -> Tuple[float, float]:
    """
    McNemar test comparing two classifiers.

    Parameters
    ----------
    correct_a, correct_b : sequences of 0/1 (1 = correct)

    Returns
    -------
    chi2 : float
    p_value : float  (two-sided, continuity-corrected)
    """
    b = sum(a == 1 and c == 0 for a, c in zip(correct_a, correct_b))
    c = sum(a == 0 and cv == 1 for a, cv in zip(correct_a, correct_b))
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0
    # Continuity-corrected McNemar
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # chi2(1) p-value approximation
    p_value = _chi2_sf(chi2, df=1)
    return float(chi2), float(p_value)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function of chi-squared distribution via regularised gamma."""
    # Using math.gammaincc (Python 3.11+) or fallback approximation
    try:
        import math
        return math.gammaincc(df / 2, x / 2)
    except AttributeError:
        # Approximation for df=1
        return 2 * _normal_sf(math.sqrt(x))


def _normal_sf(z: float) -> float:
    """Survival function of standard normal (erfc-based)."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def throughput(elapsed_s: float, n_items: int) -> float:
    """Return decisions per second."""
    if elapsed_s <= 0:
        return float("inf")
    return n_items / elapsed_s


def latency_ms(elapsed_s: float, n_items: int) -> float:
    """Return mean milliseconds per decision."""
    if n_items <= 0:
        return 0.0
    return (elapsed_s / n_items) * 1000.0


# ---------------------------------------------------------------------------
# CEIS-specific: order-flip rate & claim-only accuracy
# ---------------------------------------------------------------------------


def order_flip_rate(
    pred_normal: Sequence[str],
    pred_swapped: Sequence[str],
) -> float:
    """
    Fraction of examples where swapping claim↔evidence changes the verdict.

    Lower is better (model is symmetric / robust to order).
    """
    if len(pred_normal) != len(pred_swapped):
        raise ValueError("Both sequences must have the same length")
    flips = sum(a != b for a, b in zip(pred_normal, pred_swapped))
    return flips / max(1, len(pred_normal))


def claim_only_accuracy(
    pred_claim_only: Sequence[str],
    gold: Sequence[str],
) -> float:
    """
    Accuracy when evidence is replaced with an empty string.

    A model that cheats on claim-only signals will have higher-than-chance
    claim-only accuracy. Ideally close to 1/K = 0.333 for K=3.
    """
    return accuracy(pred_claim_only, gold)
