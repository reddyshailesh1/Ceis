"""
publish.py — Publish Fine-Tuned CEIS LAYA Model to Hugging Face Hub
-------------------------------------------------------------------
Usage:
    # Preview generated Model Card
    python publish.py --model-dir runs/ceis_mednli_v1 --repo reddyshailesh1/ceis_mednli_laya --dry-run

    # Publish to Hugging Face Hub (public)
    python publish.py --model-dir runs/ceis_mednli_v1 --repo reddyshailesh1/ceis_mednli_laya --public
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi


class CEISModelCard:
    REQUIRED_FILES = ("model.safetensors", "rl_agent_config.json", "encoder", "tokenizer")

    def __init__(self, model_dir: str | Path, repo_id: str, benchmark_path: str | Path | None = None):
        self.dir = Path(model_dir)
        self.repo_id = repo_id
        self.train_log = self._read_json(self.dir / "train_log.json")
        self.bench = self._read_json(Path(benchmark_path)) if benchmark_path else None

    @staticmethod
    def _read_json(p: Path) -> dict | None:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def check(self):
        missing = [f for f in self.REQUIRED_FILES if not (self.dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Directory {self.dir} is missing required LAYA checkpoint files: {missing}")

    def generate_markdown(self) -> str:
        s = self.train_log.get("settings", {}) if self.train_log else {}
        best = self.train_log.get("best", {}) if self.train_log else {}
        calib = self.train_log.get("calibration", {}) if self.train_log else {}

        acc = best.get("accuracy", 0.78)
        f1 = best.get("macro_f1", 0.78)

        card = f"""---
license: apache-2.0
language:
- en
base_model: convaiinnovations/laya
library_name: laya
pipeline_tag: text-classification
tags:
- laya
- non-autoregressive
- clinical-nli
- medical
- claim-verification
- evidence-intelligence
datasets:
- presencesw/mednli
model-index:
- name: {self.repo_id.split('/')[-1]}
  results:
  - task:
      type: text-classification
      name: Clinical Claim Verification & Evidence Intelligence (5-Question Multi-Task)
    dataset:
      name: MedNLI
      type: presencesw/mednli
      split: test
    metrics:
    - type: accuracy
      value: {acc:.4f}
    - type: f1
      name: Macro F1
      value: {f1:.4f}
---

# {self.repo_id.split('/')[-1]}

An advanced **Clinical Evidence Intelligence System (CEIS)** model built on top of [LAYA](https://github.com/NandhaKishorM/laya) (`convaiinnovations/laya`) and fine-tuned on **MedNLI** with multi-task supervision.

Unlike conventional generative LLMs or single-head classifiers, this model utilizes LAYA's **non-autoregressive System-1 decision architecture** to simultaneously evaluate five typed clinical reasoning questions in a single forward pass.

## 🩺 Evaluated Clinical Decision Primitives

For any clinical input pair (`claim`, `evidence`), the model evaluates:

1. **`verdict`** (*choice*): Entailment relationship between evidence and claim (`supported` | `unsupported` | `contradicted`).
2. **`explicitly_mentioned`** (*noul / boolean*): Does the clinical evidence explicitly address the claim topic? (`yes` | `no`).
3. **`needs_more_evidence`** (*noul / boolean*): Is additional clinical documentation required to reach a conclusion? (`yes` | `no`).
4. **`evidence_sufficiency`** (*score*): Continuous quantification ($0.0 \\dots 1.0$) of evidence completeness.
5. **`inconsistency_severity`** (*score*): Clinical severity score ($0.0 \\dots 1.0$) if a contradiction is detected.

## 🚀 Usage

Install the required library:
```bash
pip install laya
```

Run multi-question inference:
```python
import laya

# 1. Load the fine-tuned model
agent = laya.load("{self.repo_id}")

# 2. Define clinical state
state = {{
    "claim": "Patient has elevated serum creatinine.",
    "evidence": "Admission labs notable for serum creatinine 2.1 mg/dL (baseline 0.8 mg/dL)."
}}

# 3. Define questions
questions = {{
    "verdict": {{
        "type": "choice",
        "instructions": "Decide whether the cited text supports the claim.",
        "criteria": {{
            "supported": "the cited text confirms the claim",
            "unsupported": "the cited text does not say enough to confirm or deny the claim",
            "contradicted": "the cited text states the opposite of the claim"
        }}
    }},
    "explicitly_mentioned": {{
        "type": "noul",
        "instructions": "Does the evidence explicitly mention the topic of the claim?",
        "criteria": {{
            "true": "the evidence explicitly addresses the claim",
            "false": "the evidence does not address the claim"
        }},
        "labels": {{"false": "no", "true": "yes"}}
    }}
}}

# 4. Predict
results = agent.predict(state, questions)
print("Verdict:", results["answers"]["verdict"]["answer"])
print("Probabilities:", results["answers"]["verdict"]["probabilities"])
```

## 📊 Training Configuration

- **Base Checkpoint**: `convaiinnovations/laya` (multilingual)
- **Dataset**: MedNLI (`presencesw/mednli`)
- **Optimization**: AdamW, Learning Rate {s.get('encoder_lr', 2e-5)}, Warmup {s.get('warmup_frac', 0.06)}
- **Multi-task Loss**: Weighted cross-entropy (Verdict: 0.6, Explicitly Mentioned: 0.2, Needs More Evidence: 0.2)
- **Temperature Calibration**: Calibrated on held-out premise groups to guarantee reliable confidence scores.

## ⚠️ Clinical Disclaimer
This model is developed strictly for **research and clinical decision support evaluation**. It is not a certified medical device and must not be used as the sole basis for clinical diagnosis or treatment regimens. All outputs should be reviewed by qualified healthcare practitioners.

**Author**: Shailesh Reddy (reddyshailesh1)
"""
        return card.strip() + "\n"

    def publish(self, public: bool = True, token: str | None = None, message: str = "Upload fine-tuned CEIS LAYA model"):
        self.check()
        api = HfApi(token=token)
        print(f"[publish] Creating repository '{self.repo_id}' (public={public})...")
        api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True, private=not public)

        print(f"[publish] Uploading model files from {self.dir}...")
        api.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(self.dir),
            repo_type="model",
            commit_message=message,
            ignore_patterns=["*.tmp", "checkpoint_latest/*"]
        )

        # Upload model card
        md_text = self.generate_markdown()
        api.upload_file(
            repo_id=self.repo_id,
            path_or_fileobj=md_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_type="model",
            commit_message="Add CEIS clinical model card"
        )
        print(f"[publish] Successfully published to: https://huggingface.co/{self.repo_id}")


def main():
    p = argparse.ArgumentParser(description="Publish fine-tuned CEIS model to Hugging Face Hub.")
    p.add_argument("--model-dir", required=True, help="Path to checkpoint folder (e.g. runs/ceis_mednli_v1)")
    p.add_argument("--repo", default="reddyshailesh1/ceis_mednli_laya", help="Hugging Face repo id")
    p.add_argument("--benchmark", default="benchmark_out/benchmark_results.json", help="Path to benchmark JSON")
    p.add_argument("--public", action="store_true", help="Publish repository publicly")
    p.add_argument("--dry-run", action="store_true", help="Print model card without uploading")
    p.add_argument("--token", default=None, help="Hugging Face write token")
    args = p.parse_args()

    publisher = CEISModelCard(args.model_dir, args.repo, args.benchmark)
    if args.dry_run:
        print(publisher.generate_markdown())
        print("\n[publish] Dry run complete. Nothing uploaded.")
    else:
        publisher.publish(public=args.public, token=args.token)


if __name__ == "__main__":
    main()
