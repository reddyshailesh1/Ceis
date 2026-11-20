"""
hyperopt.py  —  Optuna hyperparameter search for CEIS
------------------------------------------------------
Search space:
  encoder_lr      [1e-5, 5e-5]   (log-uniform)
  head_lr         [5e-5, 5e-4]   (log-uniform)
  mismatch_frac   [0.10, 0.30]
  label_smoothing [0.00, 0.15]

Metric: validation macro-F1 on the verdict question (3-class).
Best params saved to <out>/best_params.json.

CLI:
  python training/hyperopt.py --config config/train_config.yaml --trials 20 --out runs/hyperopt
  python training/hyperopt.py --config config/train_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from training.trainer import CEISTrainer, TrainSettings, _load_yaml

# ---------------------------------------------------------------------------
# Optuna import guard
# ---------------------------------------------------------------------------
try:
    import optuna  # type: ignore
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _objective(trial: "optuna.Trial", base_cfg: dict,
               out_root: Path, dry_run: bool) -> float:
    """
    One Optuna trial = one full CEIS training run.
    Returns val macro-F1 on verdict (higher is better).
    """
    # --- sample hyperparameters ---
    encoder_lr      = trial.suggest_float("encoder_lr", 1e-5, 5e-5, log=True)
    head_lr         = trial.suggest_float("head_lr", 5e-5, 5e-4, log=True)
    mismatch_frac   = trial.suggest_float("mismatch_frac", 0.10, 0.30)
    label_smoothing = trial.suggest_float("label_smoothing", 0.00, 0.15)

    trial_dir = out_root / f"trial_{trial.number:03d}"

    cfg = dict(base_cfg,
               out=str(trial_dir),
               encoder_lr=encoder_lr,
               head_lr=head_lr,
               mismatch_frac=mismatch_frac,
               label_smoothing=label_smoothing)

    if dry_run:
        cfg["limit"] = 100
        cfg["epochs"] = 2

    settings = TrainSettings._from_dict(cfg)

    try:
        result = CEISTrainer(settings).run()
    except Exception as exc:
        # Let Optuna mark this trial as failed and continue
        raise optuna.exceptions.TrialPruned(f"Training raised: {exc}") from exc

    # Extract primary metric: best epoch val macro_f1
    best = result.get("best") or {}
    macro_f1 = float(best.get("macro_f1", 0.0))

    # Store useful info in Optuna trial attributes
    trial.set_user_attr("macro_f1",  macro_f1)
    trial.set_user_attr("accuracy",  float(best.get("accuracy", 0.0)))
    trial.set_user_attr("nll",       float(best.get("nll", 999.0)))
    trial.set_user_attr("best_epoch", int(best.get("epoch", -1)))

    return macro_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for CEIS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, metavar="FILE",
                   help="Base YAML config (config/train_config.yaml)")
    p.add_argument("--trials", type=int, default=20,
                   help="Number of Optuna trials")
    p.add_argument("--out", default="runs/hyperopt",
                   help="Root output directory (one sub-dir per trial)")
    p.add_argument("--study-name", default="ceis_hyperopt",
                   help="Optuna study name")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL (e.g. sqlite:///runs/hyperopt/study.db). "
                        "Omit for in-memory (non-persistent).")
    p.add_argument("--dry-run", action="store_true",
                   help="Run 2 trials with limit=100 for quick testing")
    p.add_argument("--seed", type=int, default=42,
                   help="Optuna sampler seed")
    args = p.parse_args(argv)

    base_cfg = _load_yaml(args.config)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n_trials = 2 if args.dry_run else args.trials

    # Reproducible TPE sampler
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    print(f"Starting Optuna study '{args.study_name}' — {n_trials} trials")
    print(f"Output root: {out_root.resolve()}")
    if args.dry_run:
        print("DRY-RUN: 2 trials, limit=100 each")

    study.optimize(
        lambda trial: _objective(trial, base_cfg, out_root, dry_run=args.dry_run),
        n_trials=n_trials,
        catch=(Exception,),
    )

    # --- report best trial ---
    best_trial = study.best_trial
    best_params = {
        "trial":           best_trial.number,
        "macro_f1":        best_trial.value,
        "params":          best_trial.params,
        "user_attrs":      best_trial.user_attrs,
        "study_name":      args.study_name,
        "n_trials":        len(study.trials),
        "n_complete":      sum(1 for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE),
    }

    best_path = out_root / "best_params.json"
    best_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print(f"\nBest macro-F1: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    print(f"Saved to: {best_path.resolve()}")

    # --- save all trials summary ---
    all_trials = []
    for t in study.trials:
        entry = {
            "number": t.number,
            "state":  t.state.name,
            "value":  t.value,
            "params": t.params,
            "attrs":  t.user_attrs,
        }
        all_trials.append(entry)
    (out_root / "all_trials.json").write_text(
        json.dumps(all_trials, indent=2), encoding="utf-8")

    return best_params


if __name__ == "__main__":
    main()
