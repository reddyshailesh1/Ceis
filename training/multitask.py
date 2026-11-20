"""
multitask.py  —  Multi-task loss utilities for CEIS
-----------------------------------------------------
Provides:
  multitask_loss  — weighted sum of cross-entropy losses across question heads
  silver_labels   — generate yesno silver labels from verdict label tensors
"""

from __future__ import annotations

import random
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


# ---------------------------------------------------------------------------
# Multi-task weighted loss
# ---------------------------------------------------------------------------

def multitask_loss(
    logits_dict: dict[str, torch.Tensor],
    labels_dict: dict[str, torch.Tensor],
    weights_dict: dict[str, float],
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute a weighted sum of cross-entropy losses across multiple question heads.

    Args:
        logits_dict:    Maps question-id → logit tensor of shape (B, C_q).
        labels_dict:    Maps question-id → integer label tensor of shape (B,).
        weights_dict:   Maps question-id → scalar weight (should sum to 1.0).
        label_smoothing: Epsilon for label smoothing applied to every head.

    Returns:
        (total_loss, per_head_losses) where per_head_losses is a plain dict of
        {qid: float} for logging.

    Raises:
        KeyError: if logits_dict and labels_dict have mismatched keys.
    """
    if set(logits_dict) != set(labels_dict):
        raise KeyError(
            f"logits_dict keys {set(logits_dict)} != labels_dict keys {set(labels_dict)}"
        )

    total = torch.tensor(0.0, dtype=torch.float32,
                         device=next(iter(logits_dict.values())).device)
    per_head: dict[str, float] = {}

    for qid, logits in logits_dict.items():
        labels = labels_dict[qid].to(logits.device)
        weight = weights_dict.get(qid, 1.0)
        loss = F.cross_entropy(logits.float(), labels, label_smoothing=label_smoothing)
        per_head[qid] = round(float(loss.detach().cpu()), 5)
        total = total + weight * loss

    return total, per_head


# ---------------------------------------------------------------------------
# Silver label generation
# ---------------------------------------------------------------------------

def silver_labels(
    verdict_labels: torch.Tensor,
    k_verdict: int = 3,
    rng: Optional[random.Random] = None,
) -> dict[str, torch.Tensor]:
    """
    Generate yesno silver labels for ``explicitly_mentioned`` and
    ``needs_more_evidence`` from verdict label indices.

    Label mapping (CEIS convention):
        0 = supported
        1 = unsupported
        2 = contradicted

    Silver rules:
        supported    → explicitly_mentioned=yes(1), needs_more_evidence=no(0)
        contradicted → explicitly_mentioned=yes(1), needs_more_evidence=no(0)
        unsupported  → needs_more_evidence=yes(1) always;
                       explicitly_mentioned=no(0) with p=0.60, yes(1) with p=0.40

    Yesno encoding: yes → 1, no → 0

    Args:
        verdict_labels: 1-D integer tensor of verdict label indices, shape (B,).
        k_verdict:      Number of verdict classes (default 3: sup/unsup/contra).
        rng:            Optional seeded ``random.Random`` for reproducible noise
                        on the unsupported explicitly_mentioned coin-flip.
                        Defaults to the global ``random`` module.

    Returns:
        dict with keys "explicitly_mentioned" and "needs_more_evidence",
        each a 1-D integer tensor of shape (B,) on the same device.
    """
    if rng is None:
        rng = random.Random()

    device = verdict_labels.device
    B = verdict_labels.shape[0]

    mentioned     = torch.zeros(B, dtype=torch.long, device=device)
    needs_more    = torch.zeros(B, dtype=torch.long, device=device)

    supported_idx    = 0  # CEISSchema.LABELS index
    unsupported_idx  = 1
    contradicted_idx = 2

    for i in range(B):
        v = int(verdict_labels[i].item())
        if v == supported_idx or v == contradicted_idx:
            # Evidence was used → it explicitly mentions the topic
            mentioned[i] = 1   # yes
            needs_more[i] = 0  # no
        elif v == unsupported_idx:
            # Evidence didn't settle the claim → more evidence needed
            needs_more[i] = 1  # yes
            # 40 % chance the topic was at least mentioned, 60 % it wasn't
            mentioned[i] = 1 if rng.random() < 0.40 else 0
        else:
            # Unexpected label — default to unknown (mentioned=no, needs_more=yes)
            mentioned[i] = 0
            needs_more[i] = 1

    return {
        "explicitly_mentioned": mentioned,
        "needs_more_evidence":  needs_more,
    }


# ---------------------------------------------------------------------------
# Score pseudo-labels (for logging / auxiliary objectives — not in main loss)
# ---------------------------------------------------------------------------

def score_pseudo_labels(
    verdict_labels: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    Generate soft pseudo-labels (floats in [0, 1]) for the two score questions.

    evidence_sufficiency:
        supported    → 0.9  (evidence was sufficient)
        contradicted → 0.85 (evidence was sufficient to contradict)
        unsupported  → 0.3  (evidence insufficient)

    inconsistency_severity:
        contradicted → 0.8  (significant inconsistency)
        supported    → 0.0
        unsupported  → 0.1

    When ``confidence`` (a float tensor of model softmax confidence, shape (B,))
    is provided, sufficiency is scaled by it for supported/contradicted.

    Args:
        verdict_labels: 1-D integer tensor, shape (B,).
        confidence:     Optional float tensor of shape (B,) in [0, 1].

    Returns:
        dict with "evidence_sufficiency" and "inconsistency_severity",
        each a float tensor of shape (B,).
    """
    device = verdict_labels.device
    B = verdict_labels.shape[0]

    sufficiency = torch.zeros(B, dtype=torch.float32, device=device)
    severity    = torch.zeros(B, dtype=torch.float32, device=device)

    base_suf = {0: 0.9, 1: 0.3, 2: 0.85}
    base_sev = {0: 0.0, 1: 0.1, 2: 0.8}

    for i in range(B):
        v = int(verdict_labels[i].item())
        suf = base_suf.get(v, 0.5)
        sev = base_sev.get(v, 0.1)
        if confidence is not None and v in (0, 2):
            conf = float(confidence[i].item())
            suf = suf * conf + 0.5 * (1.0 - conf)
        sufficiency[i] = suf
        severity[i]    = sev

    return {
        "evidence_sufficiency":  sufficiency,
        "inconsistency_severity": severity,
    }
