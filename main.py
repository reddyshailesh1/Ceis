#!/usr/bin/env python3
"""
main.py  —  Unified CLI for the Clinical Evidence Intelligence System (CEIS)
------------------------------------------------------------------------------
Usage:
    python main.py data download|clean|split
    python main.py train    [--config ...] [--out ...] [--seed ...] [--device ...]
    python main.py hyperopt [--config ...] [--trials N] [--out ...]
    python main.py eval     benchmark|calibration|ood|report
    python main.py infer    --claim "..." --evidence "..."
    python main.py serve    [--port 8000]
    python main.py ui       [--port 8501]
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_data(argv):
    """Dataset pipeline: download | clean | split"""
    p = argparse.ArgumentParser(description="Dataset pipeline")
    p.add_argument("action", choices=["download", "clean", "split", "all"])
    p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    p.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    p.add_argument("--splits", default=str(ROOT / "data" / "splits"))
    ns = p.parse_args(argv)

    if ns.action in ("download", "all"):
        from data.download import DataDownloader
        DataDownloader(ns.raw).run()

    if ns.action in ("clean", "all"):
        from data.clean import DataCleaner
        DataCleaner(ns.raw, ns.processed).run()

    if ns.action in ("split", "all"):
        from data.split import DataSplitter
        DataSplitter(ns.processed, ns.splits).run()


def cmd_train(argv):
    """Fine-tune CEIS model."""
    from training.trainer import CEISTrainer, CEISTrainSettings

    p = argparse.ArgumentParser(description="Fine-tune CEIS on MedNLI.")
    p.add_argument("--config", default=str(ROOT / "config" / "train_config.yaml"))
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    ns = p.parse_args(argv)

    settings = CEISTrainSettings.parse([
        "--config", ns.config,
        *(["--out", ns.out] if ns.out else []),
        *(["--seed", str(ns.seed)] if ns.seed is not None else []),
        *(["--device", ns.device] if ns.device else []),
        *(["--limit", str(ns.limit)] if ns.limit else []),
        *(["--epochs", str(ns.epochs)] if ns.epochs else []),
    ])
    CEISTrainer(settings).run()


def cmd_hyperopt(argv):
    """Hyperparameter search with Optuna."""
    from training.hyperopt import run_hyperopt

    p = argparse.ArgumentParser(description="Optuna hyperparameter search.")
    p.add_argument("--config", default=str(ROOT / "config" / "train_config.yaml"))
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--out", default=str(ROOT / "runs" / "hyperopt"))
    p.add_argument("--dry-run", action="store_true")
    ns = p.parse_args(argv)
    run_hyperopt(ns.config, ns.trials, ns.out, dry_run=ns.dry_run)


def cmd_eval(argv):
    """Evaluation suite: benchmark | calibration | ood | report | all"""
    p = argparse.ArgumentParser(description="Evaluation suite.")
    p.add_argument("action", choices=["benchmark", "calibration", "ood", "report", "all"])
    p.add_argument("--config", default=str(ROOT / "config" / "eval_config.yaml"))
    p.add_argument("--out", default=str(ROOT / "benchmark_out"))
    ns = p.parse_args(argv)

    if ns.action in ("benchmark", "all"):
        from evaluation.benchmark import Benchmark, load_eval_config
        cfg = load_eval_config(ns.config)
        Benchmark(cfg).run()

    if ns.action in ("calibration", "all"):
        from evaluation.calibration import CalibrationAnalyzer, load_eval_config
        cfg = load_eval_config(ns.config)
        CalibrationAnalyzer(cfg).run()

    if ns.action in ("ood", "all"):
        from evaluation.ood_eval import OODEvaluator, load_eval_config
        cfg = load_eval_config(ns.config)
        OODEvaluator(cfg).run()

    if ns.action in ("report", "all"):
        from evaluation.report import ReportGenerator
        ReportGenerator(ns.out, ROOT / "reports").run()


def cmd_infer(argv):
    """Single-example inference."""
    p = argparse.ArgumentParser(description="Clinical evidence inference.")
    p.add_argument("--model", default="pavanmantha/mednli_laya")
    p.add_argument("--claim", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", action="store_true", dest="as_json")
    ns = p.parse_args(argv)

    from inference.pipeline import ClinicalEvidencePipeline
    pipe = ClinicalEvidencePipeline(ns.model, device=ns.device)
    result = pipe.verify_claim(ns.claim, ns.evidence)

    if ns.as_json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print("\n" + "─" * 60)
        badge = {"supported": "✓ SUPPORTED", "unsupported": "~ UNSUPPORTED",
                 "contradicted": "✗ CONTRADICTED"}.get(result.verdict, result.verdict.upper())
        print(f"  Verdict  : {badge}  ({result.confidence:.1%} confidence)")
        print(f"  Explicit : {'yes' if result.explicitly_mentioned else 'no'}"
              f"   Needs more: {'yes' if result.needs_more_evidence else 'no'}")
        print(f"  Sufficiency: {result.evidence_sufficiency:.2f}"
              f"   Severity: {result.inconsistency_severity:.2f}")
        if result.human_review_recommended:
            print("  ⚠  Human review recommended")
        print(f"\n  Reasoning:\n  {result.reasoning}")
        if result.supporting_spans:
            print(f"\n  Supporting spans:")
            for s in result.supporting_spans:
                print(f"    + {s}")
        if result.contradicting_spans:
            print(f"\n  Contradicting spans:")
            for s in result.contradicting_spans:
                print(f"    - {s}")
        print("─" * 60 + "\n")


def cmd_serve(argv):
    """Start the FastAPI backend."""
    p = argparse.ArgumentParser(description="Start FastAPI backend.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    ns = p.parse_args(argv)

    import uvicorn
    uvicorn.run("app.api:app", host=ns.host, port=ns.port, reload=ns.reload)


def cmd_ui(argv):
    """Start the Streamlit frontend."""
    import subprocess
    p = argparse.ArgumentParser(description="Start Streamlit UI.")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--backend", default="http://localhost:8000")
    ns = p.parse_args(argv)

    ui_path = str(ROOT / "app" / "frontend" / "streamlit_app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port", str(ns.port),
        "--",
        "--backend", ns.backend,
    ])


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

COMMANDS = {
    "data":    (cmd_data,    "Dataset pipeline (download/clean/split)"),
    "train":   (cmd_train,   "Fine-tune CEIS model"),
    "hyperopt":(cmd_hyperopt,"Optuna hyperparameter search"),
    "eval":    (cmd_eval,    "Evaluation suite (benchmark/calibration/ood/report)"),
    "infer":   (cmd_infer,   "Single-example clinical inference"),
    "serve":   (cmd_serve,   "Start FastAPI backend"),
    "ui":      (cmd_ui,      "Start Streamlit frontend"),
}


def main():
    p = argparse.ArgumentParser(
        description="CEIS — Clinical Evidence Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {cmd:10s} {desc}" for cmd, (_, desc) in COMMANDS.items()))
    p.add_argument("command", choices=list(COMMANDS))
    p.add_argument("args", nargs=argparse.REMAINDER)
    ns = p.parse_args()
    COMMANDS[ns.command][0](ns.args)


if __name__ == "__main__":
    main()
