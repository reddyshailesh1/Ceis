"""
trainer.py  —  CEIS multi-task fine-tuner
------------------------------------------
Extends the ClaimCheckTrainer pattern from finetune_laya/src/claimcheck_train.py
to multi-task learning with 5 questions per (claim, evidence) pair.

Starting checkpoint : pavanmantha/mednli_laya  (already MedNLI fine-tuned)
Data                : MedNLI via data/splits/ or HuggingFace
Multi-task loss     :
    verdict               weight 0.6  (choice head, 3-way)
    explicitly_mentioned  weight 0.2  (yesno head, 2-way, silver labels)
    needs_more_evidence   weight 0.2  (yesno head, 2-way, silver labels)

CLI:
    python training/trainer.py --out runs/ceis_seed0 --config config/train_config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    class _DummyTorch:
        @staticmethod
        def no_grad():
            return lambda fn: fn
    torch = _DummyTorch()
    F = None

# ---------------------------------------------------------------------------
# laya imports (same surface as claimcheck_train.py)
# ---------------------------------------------------------------------------
try:
    from laya.calibrate import fit_temperature_map
    from laya.common import QTYPES, build_sequence, collate_items
    from laya.train import encode_state, load_checkpoint, make_item, save_checkpoint, to_internal
except ImportError:
    fit_temperature_map = None
    QTYPES = {"choice": 0, "yesno": 1, "score": 2}
    build_sequence = collate_items = encode_state = load_checkpoint = make_item = save_checkpoint = to_internal = None

from training.schema import CEISSchema
from training.multitask import multitask_loss, silver_labels

# ---------------------------------------------------------------------------
# Thin YAML helper (copied from claimcheck_config.py so we're self-contained)
# ---------------------------------------------------------------------------

def _load_yaml(path: "str | Path") -> dict:
    import re
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = [re.sub(r"\s*#.*$", "", ln) for ln in text.splitlines()]
    try:
        import yaml  # type: ignore
        return yaml.safe_load("\n".join(lines)) or {}
    except ImportError:
        pass
    result: dict = {}
    for ln in lines:
        ln = ln.strip()
        if not ln or ":" not in ln:
            continue
        key, _, val = ln.partition(":")
        key, val = key.strip(), val.strip()
        if val.lower() == "null":
            result[key] = None
        elif val.lower() == "true":
            result[key] = True
        elif val.lower() == "false":
            result[key] = False
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val.strip('"').strip("'")
    return result


# ---------------------------------------------------------------------------
# TrainSettings  (CEIS-extended)
# ---------------------------------------------------------------------------

@dataclass
class TrainSettings:
    out: str
    # Starting checkpoint — the already fine-tuned MedNLI model
    base: str = "pavanmantha/mednli_laya"
    # Data
    repo: str = "presencesw/mednli"
    data_dir: Optional[str] = None
    deid: str = "keep"
    # Hardware
    device: str = "cuda:0"
    # Reproducibility
    seed: int = 0
    # Optimisation
    epochs: int = 6
    batch: int = 32
    grad_accum: int = 2
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    shuffle_prob: float = 1.0
    mismatch_frac: float = 0.15
    patience: int = 2
    calib_items: int = 400
    eval_batch: int = 128
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    freeze_embeddings: bool = False
    grad_checkpointing: bool = False
    limit: Optional[int] = None
    # Multi-task weights
    weight_verdict: float = 0.6
    weight_mentioned: float = 0.2
    weight_needs_more: float = 0.2
    # Yesno variant to use at training time (0 = first wording)
    yesno_variant: int = 0

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def _from_dict(cls, d: dict) -> "TrainSettings":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, yaml_path: "str | Path | None" = None,
                  overrides: "dict | None" = None) -> "TrainSettings":
        cfg: dict[str, Any] = {}
        if yaml_path is not None:
            cfg.update(_load_yaml(yaml_path))
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        return cls._from_dict(cfg)

    @classmethod
    def add_to_parser(cls, p: argparse.ArgumentParser) -> None:
        defaults = cls(out="")
        p.add_argument("--config", default=None, metavar="FILE")
        p.add_argument("--out", required=False, default=None)
        for f in fields(cls):
            if f.name == "out":
                continue
            flag = "--" + f.name.replace("_", "-")
            default = getattr(defaults, f.name)
            if isinstance(default, bool):
                p.add_argument(flag, action="store_true", default=None)
            elif default is None:
                int_fields = {"max_len", "head_max_len", "limit"}
                p.add_argument(flag, default=None,
                               type=int if f.name in int_fields else str)
            else:
                p.add_argument(flag, default=None, type=type(default))

    @classmethod
    def parse(cls, argv=None) -> "TrainSettings":
        p = argparse.ArgumentParser(
            description="Fine-tune Laya (CEIS multi-task) starting from pavanmantha/mednli_laya.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cls.add_to_parser(p)
        ns = vars(p.parse_args(argv))
        yaml_path = ns.pop("config", None)
        base_cfg: dict[str, Any] = {}
        if yaml_path:
            base_cfg.update(_load_yaml(yaml_path))
        for k, v in ns.items():
            if v is not None:
                base_cfg[k] = v
        if "out" not in base_cfg or not base_cfg["out"]:
            p.error("--out is required (or set 'out' in your YAML config)")
        return cls._from_dict(base_cfg)


# ---------------------------------------------------------------------------
# Structured logger (self-contained copy, same API as TrainLogger)
# ---------------------------------------------------------------------------

class CEISLogger:
    SEP = "─" * 72

    def __init__(self, out_dir: "str | Path"):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.out / "training_log.jsonl", "a", encoding="utf-8")
        self._start = time.time()

    def _c(self, text: str, code: str) -> str:
        if not sys.stdout.isatty():
            return text
        codes = {"bold": "\033[1m", "cyan": "\033[36m", "green": "\033[32m",
                 "yellow": "\033[33m", "red": "\033[31m", "dim": "\033[2m"}
        return codes.get(code, "") + text + "\033[0m"

    def section(self, title: str) -> None:
        print(self._c(self.SEP, "dim"))
        print(self._c(f"  {title}", "cyan"))
        print(self._c(self.SEP, "dim"), flush=True)

    def info(self, msg: str, **kw: Any) -> None:
        self._emit("info", msg, **kw)
        print(f"  {msg}", flush=True)

    def step(self, epoch: int, step: int, total: int, loss: float,
             per_head: dict | None = None) -> None:
        pct = int(100 * step / total)
        bar = ("█" * (pct // 5)).ljust(20)
        head_str = ""
        if per_head:
            head_str = "  " + "  ".join(f"{k}={v:.4f}" for k, v in per_head.items())
        line = (f"  epoch {epoch}  [{bar}] {pct:3d}%  "
                f"step {step}/{total}  loss {loss:.4f}{head_str}")
        print(self._c(line, "dim"), end="\r", flush=True)

    def epoch(self, epoch: int, metrics: dict, train_loss: float,
              is_best: bool, elapsed_min: float) -> None:
        print()
        tag = self._c(" ← best", "green") if is_best else ""
        line = (f"  epoch {epoch:2d}  "
                f"acc {metrics['accuracy']:.4f}  "
                f"f1 {metrics['macro_f1']:.4f}  "
                f"nll {metrics['nll']:.4f}  "
                f"loss {train_loss:.4f}  "
                f"[{elapsed_min:.1f} min]{tag}")
        print(line, flush=True)
        self._emit("epoch", f"epoch {epoch}", epoch=epoch,
                   train_loss=round(train_loss, 4), **metrics,
                   elapsed_min=round(elapsed_min, 2), is_best=is_best)

    def warning(self, msg: str) -> None:
        print(self._c(f"  ⚠  {msg}", "yellow"), flush=True)
        self._emit("warning", msg)

    def early_stop(self, patience: int) -> None:
        msg = f"No improvement for {patience} epochs — stopping early."
        print(self._c(f"  ✗  {msg}", "red"), flush=True)
        self._emit("early_stop", msg)

    def final(self, best: dict, before: dict, out_dir: Path) -> None:
        self.section("Done")
        lines = [
            f"  Best epoch : {best['epoch']}",
            f"  Accuracy   : {best['accuracy']:.4f}  (before {before['accuracy']:.4f},"
            f"  Δ {best['accuracy'] - before['accuracy']:+.4f})",
            f"  Macro-F1   : {best['macro_f1']:.4f}",
            f"  NLL        : {best['nll']:.4f}",
            f"  Checkpoint : {out_dir.resolve()}",
        ]
        for ln in lines:
            print(self._c(ln, "bold"), flush=True)
        self._emit("final", "training complete", **best)

    def close(self) -> None:
        self._jsonl.flush()
        self._jsonl.close()

    def _emit(self, kind: str, msg: str, **kw: Any) -> None:
        record = {"kind": kind, "msg": msg,
                  "elapsed_s": round(time.time() - self._start, 1), **kw}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()


# ---------------------------------------------------------------------------
# Multi-task Sequence Encoder
# ---------------------------------------------------------------------------

class CEISEncoder:
    """
    Tokenises (claim, evidence) pairs and builds model-ready items for all
    active question heads: verdict (choice) + yesno heads.
    """

    def __init__(self, tok, max_len: int, head_max_len: int,
                 yesno_variant: int = 0):
        self.tok = tok
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.yesno_variant = yesno_variant

        # label counts
        self.k_verdict = len(CEISSchema.LABELS)       # 3
        self.k_yesno   = 2                             # yes / no

        # laya question type tokens
        self.qtype_choice = QTYPES["choice"]
        self.qtype_yesno  = QTYPES["noul"]

        # Pre-build internal question objects for all heads
        self.verdict_questions = [
            to_internal(CEISSchema.QID_VERDICT, CEISSchema.question(v))
            for v in CEISSchema.TRAIN_VARIANTS
        ]
        self.mentioned_question = to_internal(
            CEISSchema.QID_MENTIONED,
            CEISSchema.yesno_question(CEISSchema.QID_MENTIONED, yesno_variant))
        self.needs_more_question = to_internal(
            CEISSchema.QID_NEEDS_MORE,
            CEISSchema.yesno_question(CEISSchema.QID_NEEDS_MORE, yesno_variant))

        self.n_verdict_variants = len(self.verdict_questions)
        self._check_questions()

    def _check_questions(self):
        probe = encode_state(self.tok, "probe", self.max_len)
        for i, q in enumerate(self.verdict_questions):
            _, reason = make_item(
                self.tok, q, [1.0] + [0.0] * (self.k_verdict - 1),
                probe, self.head_max_len, self.max_len)
            if reason is not None:
                raise ValueError(f"verdict variant {i} cannot be encoded: {reason}")
        for name, q in [("mentioned", self.mentioned_question),
                        ("needs_more", self.needs_more_question)]:
            _, reason = make_item(
                self.tok, q, [1.0, 0.0],
                probe, self.head_max_len, self.max_len)
            if reason is not None:
                raise ValueError(f"yesno question '{name}' cannot be encoded: {reason}")

    def examples(self, df) -> list[dict]:
        """Convert a split dataframe to a list of encoded example dicts."""
        gold = {name: i for i, name in enumerate(CEISSchema.LABELS)}
        return [
            {
                "state_ids": encode_state(
                    self.tok, CEISSchema.state(r.claim, r.evidence), self.max_len),
                "gold": gold[r.label],
                "claim": r.claim,
                "evidence": r.evidence,
                "group": r.group,
            }
            for r in df.itertuples(index=False)
        ]

    def mismatched(self, example: dict, donor: dict) -> dict:
        """Create a cross-matched (claim, evidence) pair labelled 'unsupported'."""
        state = CEISSchema.state(example["claim"], donor["evidence"])
        return {
            "state_ids": encode_state(self.tok, state, self.max_len),
            "gold": CEISSchema.label_index("unsupported"),
            "claim": example["claim"],
            "evidence": donor["evidence"],
            "group": example["group"],
        }

    def item(self, example: dict, variant: int, order=None) -> dict:
        """Build a single verdict-head training item."""
        ids, markers = build_sequence(
            self.tok, None, self.verdict_questions[variant], self.max_len,
            self.head_max_len, option_order=order,
            state_ids=example["state_ids"])
        slot = example["gold"] if order is None else order.index(example["gold"])
        return {
            "ids": ids, "markers": markers,
            "qtype": self.qtype_choice, "label": slot,
        }

    def yesno_item(self, example: dict, qid: str,
                   silver_label: int) -> dict:
        """Build a yesno-head training item with a silver label."""
        q = (self.mentioned_question if qid == CEISSchema.QID_MENTIONED
             else self.needs_more_question)
        ids, markers = build_sequence(
            self.tok, None, q, self.max_len, self.head_max_len,
            state_ids=example["state_ids"])
        return {
            "ids": ids, "markers": markers,
            "qtype": self.qtype_yesno, "label": silver_label,
        }

    def batch(self, items: list[dict]) -> dict:
        return collate_items([items], self.tok.pad_token_id)


# ---------------------------------------------------------------------------
# CEIS Trainer
# ---------------------------------------------------------------------------

class CEISTrainer:
    COLLAPSE_LOGIT_RANGE = 0.05

    def __init__(self, settings: TrainSettings):
        self.s = settings
        self.device = torch.device(settings.device)
        self.out = Path(settings.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = CEISLogger(self.out)

        # Seeding
        self.rng = random.Random(settings.seed)
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)

        # Load from the already fine-tuned MedNLI checkpoint
        self.log.section("Loading checkpoint")
        self.log.info(f"base checkpoint: {settings.base}")
        self.model, self.tok, self.cfg = load_checkpoint(settings.base)
        self.max_len = settings.max_len or self.cfg.get("max_len", 512)
        self.head_max_len = settings.head_max_len or self.cfg.get("head_max_len", 192)
        self.enc = CEISEncoder(self.tok, self.max_len, self.head_max_len,
                               settings.yesno_variant)
        self.n_variants = self.enc.n_verdict_variants
        self.k = self.enc.k_verdict
        self.amp_dtype, self.scaler = self._amp()

        # Multi-task weights
        self.mt_weights = {
            CEISSchema.QID_VERDICT:    settings.weight_verdict,
            CEISSchema.QID_MENTIONED:  settings.weight_mentioned,
            CEISSchema.QID_NEEDS_MORE: settings.weight_needs_more,
        }

        self.data_stats: dict = {}
        self.history: list = []

    # ------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------

    def _amp(self):
        if self.device.type != "cuda":
            return None, None
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        return torch.float16, torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------
    # Data loading  (mirrors claimcheck_train.py _load_data)
    # ------------------------------------------------------------------

    def _load_data(self):
        self.log.section("Loading data")
        # Try local splits first (produced by data/split.py)
        splits_dir = Path(self.s.data_dir) if self.s.data_dir else (_ROOT / "data" / "splits")

        if splits_dir.exists() and (splits_dir / "mednli_train.csv").exists():
            import pandas as pd
            train_df = pd.read_csv(splits_dir / "mednli_train.csv")
            val_df   = pd.read_csv(splits_dir / "mednli_val.csv")
            for col in ("claim", "evidence", "label", "group"):
                if col not in train_df.columns:
                    raise ValueError(f"Missing column '{col}' in mednli_train.csv")
            if train_df["label"].iloc[0] in CEISSchema.FROM_NLI:
                train_df["label"] = train_df["label"].map(CEISSchema.FROM_NLI)
                val_df["label"]   = val_df["label"].map(CEISSchema.FROM_NLI)
        else:
            # Fall back to self-contained HuggingFace download
            import re
            import pandas as pd
            from huggingface_hub import hf_hub_download
            tr_p = hf_hub_download(self.s.repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
            va_p = hf_hub_download(self.s.repo, "data/validation-00000-of-00001.parquet", repo_type="dataset")
            raw_tr = pd.read_parquet(tr_p)
            raw_va = pd.read_parquet(va_p)
            deid_re = re.compile(r"\[\*\*.*?\*\*\]")
            def _clean(df):
                ev = df["sentence1"].astype(str).str.strip()
                if self.s.deid == "mask":
                    ev = ev.map(lambda t: deid_re.sub("[redacted]", t))
                cdf = pd.DataFrame({
                    "claim": df["sentence2"].astype(str).str.strip(),
                    "evidence": ev,
                    "label": df["gold_label"].map(CEISSchema.FROM_NLI),
                    "group": df["sentence1"].astype(str),
                })
                n_lbls = cdf.groupby(["claim", "evidence"])["label"].transform("nunique")
                return cdf[n_lbls == 1].drop_duplicates(["claim", "evidence"]).reset_index(drop=True)
            train_df = _clean(raw_tr)
            val_df = _clean(raw_va)

        if self.s.limit:
            train_df = train_df.head(self.s.limit)
            val_df   = val_df.head(max(30, self.s.limit // 4))

        fit_df, calib_df = self._split_calibration(train_df)
        self.train = self.enc.examples(fit_df)
        self.calib = self.enc.examples(calib_df)
        self.val   = self.enc.examples(val_df)
        self.data_stats.update(
            train_items=len(self.train),
            calibration_items=len(self.calib),
            validation_items=len(self.val),
            mismatch_frac=self.s.mismatch_frac,
        )
        self.log.info("data loaded", **self.data_stats)

    def _split_calibration(self, df):
        groups = sorted(df["group"].unique())
        random.Random(self.s.seed + 9173).shuffle(groups)
        sizes = df.groupby("group").size()
        target = min(self.s.calib_items, len(df) // 10)
        held, n = set(), 0
        for g in groups:
            if n >= target:
                break
            held.add(g)
            n += int(sizes[g])
        mask = df["group"].isin(held)
        return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Optimiser / scheduler  (mirrors claimcheck_train.py)
    # ------------------------------------------------------------------

    def _configure(self):
        m, s = self.model, self.s
        if s.freeze_embeddings:
            for p in m.encoder.get_input_embeddings().parameters():
                p.requires_grad_(False)
        if s.grad_checkpointing and hasattr(m.encoder, "gradient_checkpointing_enable"):
            m.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            m.head_checkpointing = True
        m.to(self.device)

        enc_params  = [p for n, p in m.named_parameters()
                       if n.startswith("encoder.") and p.requires_grad]
        head_params = [p for n, p in m.named_parameters()
                       if not n.startswith("encoder.") and p.requires_grad]
        self.params = enc_params + head_params
        self.optimizer = torch.optim.AdamW(
            [{"params": enc_params, "lr": s.encoder_lr},
             {"params": head_params, "lr": s.head_lr}],
            weight_decay=s.weight_decay)

        self.steps_per_epoch = math.ceil(len(self.train) / s.batch)
        self.total_updates   = math.ceil(self.steps_per_epoch / s.grad_accum) * s.epochs
        warm = max(1, int(self.total_updates * s.warmup_frac))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            [self._schedule(warm, s.min_lr / s.encoder_lr),
             self._schedule(warm, s.min_lr / s.head_lr)])

    def _schedule(self, warm: int, floor: float):
        total = self.total_updates

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            progress = min(1.0, (step - warm) / max(1, total - warm))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return factor

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _logits(self, batch) -> torch.Tensor:
        args = [batch[name].to(self.device) for name in
                ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        if self.amp_dtype is None:
            logits, _ = self.model(*args)
        else:
            with torch.autocast("cuda", dtype=self.amp_dtype):
                logits, _ = self.model(*args)
        return logits.float()

    # ------------------------------------------------------------------
    # Sampling helpers  (mirrors claimcheck_train.py)
    # ------------------------------------------------------------------

    def _draw_order(self):
        if self.rng.random() >= self.s.shuffle_prob:
            return None
        order = list(range(self.k))
        self.rng.shuffle(order)
        return order

    def _pick(self, index: int) -> dict:
        example = self.train[index]
        if self.rng.random() >= self.s.mismatch_frac:
            return example
        for _ in range(20):
            donor = self.train[self.rng.randrange(len(self.train))]
            if donor["group"] != example["group"]:
                return self.enc.mismatched(example, donor)
        return example

    # ------------------------------------------------------------------
    # Multi-task training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> tuple[float, dict]:
        s = self.s
        self.model.train()
        order = list(range(len(self.train)))
        self.rng.shuffle(order)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, n_steps = 0.0, 0
        total_per_head: dict[str, float] = {k: 0.0 for k in self.mt_weights}

        for start in range(0, len(order), s.batch):
            batch_examples = [self._pick(i) for i in order[start: start + s.batch]]

            # --- verdict head ---
            verdict_items = [
                self.enc.item(
                    ex, self.rng.randrange(self.n_variants), self._draw_order())
                for ex in batch_examples
            ]
            verdict_batch = self.enc.batch(verdict_items)
            verdict_logits = self._logits(verdict_batch)[:, :self.k]
            verdict_labels = verdict_batch["label"].to(self.device)

            # --- generate silver labels for yesno heads ---
            silver = silver_labels(verdict_labels, k_verdict=self.k, rng=self.rng)

            # --- explicitly_mentioned head ---
            mentioned_items = [
                self.enc.yesno_item(ex, CEISSchema.QID_MENTIONED,
                                    int(silver["explicitly_mentioned"][i].item()))
                for i, ex in enumerate(batch_examples)
            ]
            mentioned_batch  = self.enc.batch(mentioned_items)
            mentioned_logits = self._logits(mentioned_batch)[:, :2]
            mentioned_labels = mentioned_batch["label"].to(self.device)

            # --- needs_more_evidence head ---
            needs_items = [
                self.enc.yesno_item(ex, CEISSchema.QID_NEEDS_MORE,
                                    int(silver["needs_more_evidence"][i].item()))
                for i, ex in enumerate(batch_examples)
            ]
            needs_batch  = self.enc.batch(needs_items)
            needs_logits = self._logits(needs_batch)[:, :2]
            needs_labels = needs_batch["label"].to(self.device)

            # --- multi-task loss ---
            combined_loss, per_head = multitask_loss(
                logits_dict={
                    CEISSchema.QID_VERDICT:    verdict_logits,
                    CEISSchema.QID_MENTIONED:  mentioned_logits,
                    CEISSchema.QID_NEEDS_MORE: needs_logits,
                },
                labels_dict={
                    CEISSchema.QID_VERDICT:    verdict_labels,
                    CEISSchema.QID_MENTIONED:  mentioned_labels,
                    CEISSchema.QID_NEEDS_MORE: needs_labels,
                },
                weights_dict=self.mt_weights,
                label_smoothing=s.label_smoothing,
            )

            # Gradient accumulation: weight by window size
            window_start = (n_steps // s.grad_accum) * s.grad_accum
            window = min(s.grad_accum, self.steps_per_epoch - window_start)
            scaled = combined_loss / window

            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            n_steps += 1
            total_loss += float(combined_loss.detach().cpu())
            for k, v in per_head.items():
                total_per_head[k] = total_per_head.get(k, 0.0) + v

            if n_steps % s.grad_accum == 0 or n_steps == self.steps_per_epoch:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.params, s.grad_clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if n_steps % 10 == 0 or n_steps == self.steps_per_epoch:
                self.log.step(epoch, n_steps, self.steps_per_epoch,
                              total_loss / n_steps,
                              {k: v / n_steps for k, v in total_per_head.items()})

        avg_per_head = {k: v / max(1, n_steps) for k, v in total_per_head.items()}
        return total_loss / max(1, n_steps), avg_per_head

    # ------------------------------------------------------------------
    # Evaluation  (verdict head only — this is what we checkpoint on)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score(self, examples: list[dict]):
        self.model.eval()
        logits_list, labels_list = [], []
        flat = [(ex, v) for ex in examples for v in range(self.n_variants)]
        for start in range(0, len(flat), self.s.eval_batch):
            items = [self.enc.item(ex, v) for ex, v in flat[start: start + self.s.eval_batch]]
            batch = self.enc.batch(items)
            logits_list.append(self._logits(batch)[:, :self.k].cpu())
            labels_list.append(batch["label"])
        return torch.cat(logits_list), torch.cat(labels_list)

    def _metrics(self, logits, labels) -> dict:
        pred = logits.argmax(-1)
        f1 = []
        for c in range(self.k):
            tp = int(((pred == c) & (labels == c)).sum())
            fp = int(((pred == c) & (labels != c)).sum())
            fn = int(((pred != c) & (labels == c)).sum())
            f1.append(2 * tp / max(1, 2 * tp + fp + fn))
        spread = float((logits.max(-1).values - logits.min(-1).values).mean())
        return {
            "nll":        round(float(F.cross_entropy(logits, labels)), 4),
            "accuracy":   round(float((pred == labels).float().mean()), 4),
            "macro_f1":   round(sum(f1) / self.k, 4),
            "logit_range": round(spread, 4),
        }

    # ------------------------------------------------------------------
    # Temperature calibration  (mirrors claimcheck_train.py)
    # ------------------------------------------------------------------

    def _calibrate(self):
        logits, labels = self._score(self.calib)
        records = []
        for row, gold in zip(logits.numpy(), labels.tolist()):
            target = [0.0] * self.k
            target[gold] = 1.0
            records.append((self.enc.qtype_choice, row, target, self.k))
        fitted = fit_temperature_map(records)
        temp = float(fitted["temperature"][self.enc.qtype_choice])
        scaled = logits / temp
        return temp, {
            "items": len(records),
            "temperature": round(temp, 4),
            "nll_before": round(float(F.cross_entropy(logits, labels)), 4),
            "nll_after":  round(float(F.cross_entropy(scaled, labels)), 4),
            "mean_confidence_after": round(
                float(torch.softmax(scaled, -1).max(-1).values.mean()), 4),
        }

    # ------------------------------------------------------------------
    # Checkpoint saving  (mirrors claimcheck_train.py _save)
    # ------------------------------------------------------------------

    def _save(self, epoch: int, val: dict, base_val: dict):
        temp, calibration = self._calibrate()
        temps = [float(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        temps[self.enc.qtype_choice] = temp
        buckets = {k: v for k, v in (self.cfg.get("temperature_by_options") or {}).items()
                   if not k.startswith("choice:")}
        cfg = dict(self.cfg, max_len=self.max_len, head_max_len=self.head_max_len,
                   fine_tuned=True, temperature=temps)
        cfg.pop("temperature_by_options", None)
        if buckets:
            cfg["temperature_by_options"] = buckets
        cfg["training"] = dict(cfg.get("training") or {}, ceis={
            "settings":          asdict(self.s),
            "best_epoch":        epoch,
            "validation_before": base_val,
            "validation":        val,
            "calibration":       calibration,
            "data":              self.data_stats,
            "labels":            list(CEISSchema.LABELS),
            "multitask_weights": self.mt_weights,
        })
        save_checkpoint(self.model, self.tok, cfg, str(self.out))
        # Persist question schemas for inference
        questions = {
            "verdict_benchmark": {
                CEISSchema.QID_VERDICT: CEISSchema.question(CEISSchema.BENCH_VARIANT)
            },
            "verdict_train_variants": [
                {CEISSchema.QID_VERDICT: CEISSchema.question(v)}
                for v in CEISSchema.TRAIN_VARIANTS
            ],
            "yesno": {
                CEISSchema.QID_MENTIONED:  CEISSchema.yesno_question(CEISSchema.QID_MENTIONED),
                CEISSchema.QID_NEEDS_MORE: CEISSchema.yesno_question(CEISSchema.QID_NEEDS_MORE),
            },
            "score": {
                CEISSchema.QID_SUFFICIENCY: CEISSchema.score_question(CEISSchema.QID_SUFFICIENCY),
                CEISSchema.QID_SEVERITY:    CEISSchema.score_question(CEISSchema.QID_SEVERITY),
            },
        }
        (self.out / "questions.json").write_text(
            json.dumps(questions, indent=2), encoding="utf-8")
        return calibration

    # ------------------------------------------------------------------
    # Main training loop  (same structure as claimcheck_train.py run())
    # ------------------------------------------------------------------

    def run(self) -> dict:
        s = self.s
        self._load_data()
        self._configure()

        self.log.section("Training budget")
        self.log.info(
            f"  {self.total_updates} updates  ·  {s.epochs} epochs  "
            f"·  effective batch {s.batch * s.grad_accum}  ·  AMP {self.amp_dtype}")
        self.log.info("  multitask weights  " + json.dumps(self.mt_weights))
        self.log.info("  data  " + json.dumps(self.data_stats))

        base_val = self._metrics(*self._score(self.val))
        self.log.info("validation BEFORE training  " + json.dumps(base_val))

        best, stale, calibration = None, 0, None
        started = time.time()

        self.log.section("Epochs")
        for epoch in range(1, s.epochs + 1):
            train_loss, per_head = self._train_epoch(epoch)
            val = self._metrics(*self._score(self.val))
            elapsed = (time.time() - started) / 60
            improved = best is None or val["nll"] < best["nll"] - 1e-4

            entry = dict(val, epoch=epoch, train_loss=round(train_loss, 4),
                         per_head=per_head, minutes=round(elapsed, 2))
            self.history.append(entry)
            self.log.epoch(epoch, val, train_loss, improved, elapsed)

            if val["logit_range"] < self.COLLAPSE_LOGIT_RANGE:
                self.log.warning(
                    "logits are near-constant — the head may have collapsed. "
                    "Try another seed or more epochs.")

            if improved:
                best, stale = entry, 0
                calibration = self._save(epoch, val, base_val)
            else:
                stale += 1
                if stale >= s.patience:
                    self.log.early_stop(s.patience)
                    break

        result = {
            "settings":          asdict(s),
            "data":              self.data_stats,
            "validation_before": base_val,
            "best":              best,
            "calibration":       calibration,
            "history":           self.history,
        }
        (self.out / "train_log.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        self.log.final(best, base_val, self.out)
        self.log.close()
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = TrainSettings.parse()
    CEISTrainer(settings).run()


# ---------------------------------------------------------------------------
# Compatibility alias for main.py
# ---------------------------------------------------------------------------
CEISTrainSettings = TrainSettings
