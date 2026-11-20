# Clinical Evidence Intelligence System (CEIS) — Research Report
**Authors**: Pavan Kumar Mantha & Clinical Evidence Intelligence Lab

## Abstract
We present the **Clinical Evidence Intelligence System (CEIS)**, an advanced medical reasoning framework extending LAYA's non-autoregressive decision engine. By transitioning from single-label inference to a typed, multi-task decision architecture (choice, score, and yes/no primitives), CEIS verifies clinical claims against electronic health records, detects subtle document inconsistencies, quantifies evidence sufficiency, and flags clinical risk for clinician review.

## 1 · Experimental Comparison

| Model | Accuracy | Macro F1 | Brier Score | ECE (15 bins) | Latency (ms) | Throughput (dec/s) |
|---|---|---|---|---|---|---|
| **Base LAYA** | 0.3667 | 0.3004 | 0.6945 | 0.2570 | 0.04 | 28352.7 |
| **MedNLI Fine-Tuned LAYA** | 0.3000 | 0.3038 | 0.9370 | 0.5040 | 0.04 | 26438.7 |
| **CEIS Multi-Task Fine-Tuned** | 0.3000 | 0.3038 | 1.1014 | 0.5877 | 0.04 | 28267.2 |

## 2 · Out-of-Domain Generalization (PubMedQA)

To evaluate robustness against domain shift from clinical progress notes to biomedical research abstracts, models were evaluated on the PubMedQA test partition.

*OOD results will be populated when `python main.py eval ood` is executed.*

## 3 · Temperature Calibration Analysis

*Calibration metrics populated upon running `python main.py eval calibration`.*

## 4 · Architectural Strengths & Ethical Safeguards

1. **Non-Autoregressive Decision Primitives**: By directly optimizing classification and scoring boundaries rather than token generation, CEIS eliminates generative hallucinations and token drift.
2. **Explainable Reasoning**: Rather than generating open-ended justifications, the system constructs deterministic, traceable explanations directly from verified decision head outputs.
3. **Human-in-the-Loop Protocol**: Claims categorized as `contradicted`, low confidence (< 0.70), or high severity (> 0.60) trigger mandatory human review warnings.
4. **Clinical Disclaimer**: CEIS is strictly a clinical decision support research tool and does not provide autonomous patient diagnoses or treatment regimens.