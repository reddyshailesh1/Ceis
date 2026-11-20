"""
report.py — Comprehensive Research & Application Report Generator
-----------------------------------------------------------------
Aggregates:
  1. benchmark_results.json
  2. calibration_results.json
  3. ood_results.json

Produces:
  - reports/research_report.md  (comprehensive clinical NLP report)
  - reports/metrics_summary.csv (flat CSV for tabular comparison)
  - reports/dashboard_summary.json (for frontend app ingestion)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


class ReportGenerator:
    """Aggregates benchmark artifacts into publishable research reports."""

    def __init__(self, benchmark_dir: str | Path, out_dir: str | Path):
        self.bench_dir = Path(benchmark_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_json(self, filename: str) -> Optional[dict]:
        path = self.bench_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def run(self):
        bench = self._load_json("benchmark_results.json")
        calib = self._load_json("calibration_results.json")
        ood = self._load_json("ood_results.json")

        if bench is None:
            print(f"[report] No benchmark_results.json found in {self.bench_dir}. Running benchmark first...")
            from evaluation.benchmark import Benchmark
            bench = Benchmark().run()

        # Generate CSV
        csv_path = self.out_dir / "metrics_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_id", "model_name", "accuracy", "ci95_low", "ci95_high",
                "macro_f1", "brier", "ece", "order_flip_rate", "claim_only_acc",
                "latency_ms", "throughput"
            ])
            for m in bench.get("models", []):
                ci = m.get("accuracy_ci95", [0, 0])
                writer.writerow([
                    m.get("id"), m.get("name"), m.get("accuracy"),
                    ci[0], ci[1], m.get("macro_f1"), m.get("brier"),
                    m.get("ece"), m.get("order_flip_rate"), m.get("claim_only_acc"),
                    m.get("latency_ms"), m.get("throughput_dec_per_sec"),
                ])
        print(f"[report] Generated CSV summary: {csv_path}")

        # Generate Dashboard JSON
        dashboard_summary = {
            "benchmark": bench,
            "calibration": calib,
            "ood": ood,
            "summary_card": {
                "best_model": bench["models"][-1]["name"],
                "accuracy": bench["models"][-1]["accuracy"],
                "macro_f1": bench["models"][-1]["macro_f1"],
                "ece": bench["models"][-1]["ece"],
            }
        }
        json_path = self.out_dir / "dashboard_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dashboard_summary, f, indent=2)
        print(f"[report] Generated Dashboard JSON: {json_path}")

        # Generate Full Markdown Research Report
        md_path = self.out_dir / "research_report.md"
        report_md = self._build_research_report(bench, calib, ood)
        md_path.write_text(report_md, encoding="utf-8")
        print(f"[report] Generated Research Report: {md_path}")

    def _build_research_report(self, bench: dict, calib: Optional[dict], ood: Optional[dict]) -> str:
        lines = [
            "# Clinical Evidence Intelligence System (CEIS) — Research Report",
            "**Authors**: Shailesh Reddy (reddyshailesh1), Karthikeya (karthikeya3342) & Clinical Evidence Intelligence Lab",
            "",
            "## Abstract",
            "We present the **Clinical Evidence Intelligence System (CEIS)**, an advanced medical reasoning framework extending LAYA's non-autoregressive decision engine. By transitioning from single-label inference to a typed, multi-task decision architecture (choice, score, and yes/no primitives), CEIS verifies clinical claims against electronic health records, detects subtle document inconsistencies, quantifies evidence sufficiency, and flags clinical risk for clinician review.",
            "",
            "## 1 · Experimental Comparison",
            "",
            "| Model | Accuracy | Macro F1 | Brier Score | ECE (15 bins) | Latency (ms) | Throughput (dec/s) |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in bench.get("models", []):
            lines.append(
                f"| **{m['name']}** | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} | {m['latency_ms']:.2f} | {m['throughput_dec_per_sec']:.1f} |"
            )

        lines.extend([
            "",
            "## 2 · Out-of-Domain Generalization (PubMedQA)",
            "",
            "To evaluate robustness against domain shift from clinical progress notes to biomedical research abstracts, models were evaluated on the PubMedQA test partition.",
            "",
        ])
        if ood and "models" in ood:
            lines.extend([
                "| Model | PubMedQA Accuracy | PubMedQA Macro F1 | ECE |",
                "|---|---|---|---|",
            ])
            for m in ood.get("models", []):
                lines.append(f"| **{m.get('name', m.get('model'))}** | {m.get('accuracy', 0.0):.4f} | {m.get('macro_f1', 0.0):.4f} | {m.get('ece', 0.0):.4f} |")
        else:
            lines.append("*OOD results will be populated when `python main.py eval ood` is executed.*")

        lines.extend([
            "",
            "## 3 · Temperature Calibration Analysis",
            "",
        ])
        if calib and "temperature_scaling" in calib:
            t_data = calib["temperature_scaling"]
            pre_data = calib.get("pre_temperature", {})
            lines.extend([
                f"- **Optimal Temperature parameter (T)**: {t_data.get('optimal_T', 1.0):.4f}",
                f"- **Pre-calibration ECE**: {pre_data.get('ece', 0.0):.4f} → **Post-calibration ECE**: {t_data.get('post_ece', 0.0):.4f}",
                f"- **Pre-calibration Brier**: {pre_data.get('brier_score', 0.0):.4f} → **Post-calibration Brier**: {t_data.get('post_brier_score', 0.0):.4f}",
            ])
        else:
            lines.append("*Calibration metrics populated upon running `python main.py eval calibration`.*")

        lines.extend([
            "",
            "## 4 · Architectural Strengths & Ethical Safeguards",
            "",
            "1. **Non-Autoregressive Decision Primitives**: By directly optimizing classification and scoring boundaries rather than token generation, CEIS eliminates generative hallucinations and token drift.",
            "2. **Explainable Reasoning**: Rather than generating open-ended justifications, the system constructs deterministic, traceable explanations directly from verified decision head outputs.",
            "3. **Human-in-the-Loop Protocol**: Claims categorized as `contradicted`, low confidence (< 0.70), or high severity (> 0.60) trigger mandatory human review warnings.",
            "4. **Clinical Disclaimer**: CEIS is strictly a clinical decision support research tool and does not provide autonomous patient diagnoses or treatment regimens.",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate CEIS Research Report.")
    p.add_argument("--benchmark", default="benchmark_out")
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    ReportGenerator(args.benchmark, args.out).run()
