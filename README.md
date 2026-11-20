# Advanced Clinical Evidence Intelligence System (CEIS)

> Built on [`pavanmantha/mednli_laya`](https://huggingface.co/pavanmantha/mednli_laya) · Extends LAYA's non-autoregressive decision architecture for multi-task clinical NLI.

> ⚠️ **Research use only. Not a medical device. Do not use for clinical decisions.**

---

## What it does

Given a **clinical claim** and **clinical evidence**, CEIS returns 5 typed decisions in one forward pass:

| Decision | Type | Description |
|---|---|---|
| `verdict` | 3-way choice | supported / unsupported / contradicted |
| `explicitly_mentioned` | yes/no | Does the evidence explicitly mention the claim topic? |
| `needs_more_evidence` | yes/no | Is additional evidence needed to settle the claim? |
| `evidence_sufficiency` | score 0–1 | How complete is the available evidence? |
| `inconsistency_severity` | score 0–1 | If contradicted, how clinically significant is the inconsistency? |

---

## Project layout

```
ceis/
├── config/           YAML configs for training, evaluation, app
├── data/             Dataset download, cleaning, splitting scripts
├── training/         Multi-task fine-tuning, hyperopt
├── evaluation/       Benchmark, calibration, OOD eval, report generation
├── inference/        ClinicalEvidencePipeline, chunker, highlighter
├── app/
│   ├── api.py        FastAPI backend
│   └── frontend/     Streamlit UI
├── notebooks/        End-to-end research notebooks
├── reports/          Generated evaluation reports
├── runs/             Fine-tuned checkpoints (gitignored)
├── main.py           Unified CLI
├── Makefile
└── requirements.txt
```

---

## Quick start

### 1. Install
```bash
cd ceis/
pip install -r requirements.txt
```

### 2. Download & prepare data
```bash
python main.py data download
python main.py data clean
python main.py data split
```

### 3. Quick smoke test (2 min)
```bash
python main.py train --limit 200 --epochs 1 --out runs/quick
```

### 4. Full fine-tuning
```bash
python main.py train --config config/train_config.yaml
```

### 5. Benchmark all 3 models
```bash
python main.py eval benchmark
```

### 6. Run the web application
```bash
# Terminal 1 — API backend
python main.py serve --port 8000

# Terminal 2 — Streamlit frontend
python main.py ui
```

Open http://localhost:8501 in your browser.

### 7. Single inference
```bash
python main.py infer \
  --claim "Patient has elevated creatinine." \
  --evidence "Labs were notable for Cr 1.7 (baseline 0.5)."
```

---

## Models compared

| Model | Description |
|---|---|
| `multilingual` | Base LAYA (no fine-tuning) |
| `pavanmantha/mednli_laya` | MedNLI fine-tuned LAYA (our starting checkpoint) |
| `runs/ceis_seed0` | CEIS multi-task fine-tuned model |

---

## Makefile reference

| Target | Action |
|---|---|
| `make install` | pip install -r requirements.txt |
| `make data` | Download + clean + split datasets |
| `make quick` | 2-minute smoke test |
| `make train` | Full CEIS fine-tuning |
| `make eval` | 3-model benchmark |
| `make serve` | FastAPI + Streamlit |
| `make notebook` | Launch Jupyter |
| `make clean` | Remove runs/ and benchmark_out/ |

---

## Limitations

- English only
- Optimized for short clinical sentences (MedNLI-length input)
- Long passages are chunked and aggregated — accuracy may drop
- yesno/score heads use silver labels derived from MedNLI verdict labels — not independently validated
- Out-of-domain performance on real EHR text is unknown
- PubMedQA label mapping (yes/no/maybe → supported/contradicted/unsupported) is imperfect

---

## Citation / acknowledgements

- **LAYA:** [`convaiinnovations/laya`](https://github.com/NandhaKishorM/laya)
- **MedNLI:** Romanov & Shivade (2018), via [`presencesw/mednli`](https://huggingface.co/datasets/presencesw/mednli)
- **PubMedQA:** Jin et al. (2019), via [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA)
- **Fine-tuned checkpoint:** [`pavanmantha/mednli_laya`](https://huggingface.co/pavanmantha/mednli_laya)

Author: Shailesh Reddy (reddyshailesh1)
