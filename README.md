# Advanced Clinical Evidence Intelligence System (CEIS)

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-reddyshailesh1%2Fceis-blue.svg)](https://huggingface.co/reddyshailesh1/ceis)
[![GitHub Repository](https://img.shields.io/badge/GitHub-reddyshailesh1%2FCeis-black.svg?logo=github)](https://github.com/reddyshailesh1/Ceis)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

**A Non-Autoregressive, Multi-Question Decision Architecture for Clinical Evidence Verification and Contradiction Detection**

[Live Hugging Face Model](https://huggingface.co/reddyshailesh1/ceis) · [GitHub Repository](https://github.com/reddyshailesh1/Ceis) · [API Documentation](#fastapi-rest-endpoints) · [Quickstart](#-quick-start)

</div>

---

> ⚠️ **CLINICAL RESEARCH DISCLAIMER**: This system is developed strictly for healthcare NLP research and clinical decision support evaluation. It is **not** a certified medical diagnostic device and does not provide autonomous medical advice or treatment decisions. All system outputs must be verified by qualified healthcare professionals.

---

## 🌟 Overview

Conventional healthcare NLP systems often rely on either ordinary single-head text classifiers or generative large language models (LLMs). Both exhibit fundamental limitations in clinical environments:
- **Single-head classifiers** only output a single label (e.g. Entailment / Contradiction / Neutral) without explaining whether the clinical topic was explicitly discussed or if the record was simply insufficient.
- **Generative LLMs** introduce hallucinations, non-deterministic phrasing, latency bottlenecks, and uncalibrated confidence scores that risk patient safety.

The **Clinical Evidence Intelligence System (CEIS)** extends [LAYA's](https://github.com/NandhaKishorM/laya) non-autoregressive System-1 decision architecture into a multi-task clinical evidence engine. For any clinical claim and evidence passage, CEIS evaluates **five typed decision primitives simultaneously in a single forward pass**:

```
                       ┌─────────────────────────┐
                       │  Clinical Claim Pair    │
                       │  (Evidence, Claim)      │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   CEIS Decision Model   │
                       │ (Fine-Tuned LAYA Arch)  │
                       └────────────┬────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│   VERDICT    │             │  EXPLICITLY  │             │  NEEDS MORE  │
│   (Choice)   │             │  MENTIONED   │             │   EVIDENCE   │
│ Supported    │             │   (Noul)     │             │    (Noul)    │
│ Unsupported  │             │ True / False │             │ True / False │
│ Contradicted │             └──────────────┘             └──────────────┘
└──────────────┘                    │                            │
       │                            ▼                            ▼
       │                     ┌──────────────┐             ┌──────────────┐
       │                     │ SUFFICIENCY  │             │   SEVERITY   │
       │                     │   (Score)    │             │   (Score)    │
       │                     │ 0.00 ─ 1.00  │             │ 0.00 ─ 1.00  │
       │                     └──────────────┘             └──────────────┘
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Clinician Audit: Calibrated Confidence, Highlighted Evidence Spans,   │
│  Templated Reasoning (Zero Hallucination), & Human Review Flag         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🤗 Hugging Face Model

The fine-tuned model checkpoint is officially published on the Hugging Face Hub:

📦 **Model Repo**: **[`reddyshailesh1/ceis`](https://huggingface.co/reddyshailesh1/ceis)**

### ⚡ 3-Line Inference with Hugging Face

```python
import laya

# 1. Load the fine-tuned model directly from Hugging Face
agent = laya.load("reddyshailesh1/ceis")

# 2. Define clinical state
state = {
    "claim": "Patient has severe acute pancreatitis.",
    "evidence": "Serum lipase was 1450 U/L and CT abdomen showed acute pancreatitis."
}

# 3. Define decision question
question = {
    "verdict": {
        "type": "choice",
        "instructions": "Decide whether the cited text supports the claim.",
        "criteria": {
            "supported": "the cited text confirms the claim",
            "unsupported": "the cited text does not say enough to confirm or deny the claim",
            "contradicted": "the cited text states the opposite of the claim"
        }
    }
}

# 4. Predict in a single forward pass
result = agent.predict(state, question)
print("Decision:", result["answers"]["verdict"]["choice"])
print("Confidence:", result["answers"]["verdict"]["probabilities"])
```

---

## 🔬 Multi-Question Decision Primitives

| Question | Primitive Type | Possible Outputs | Clinical Purpose |
|---|---|---|---|
| **`verdict`** | `choice` | `supported` \| `unsupported` \| `contradicted` | Core clinical claim verification and entailment determination. |
| **`explicitly_mentioned`** | `noul` (boolean) | `true` \| `false` | Distinguishes whether the claim topic is explicitly addressed in the clinical note or entirely unmentioned. |
| **`needs_more_evidence`** | `noul` (boolean) | `true` \| `false` | Determines whether the available evidence is sufficient or if additional diagnostic notes are required. |
| **`evidence_sufficiency`** | `score` | Continuous index ($0.00 \dots 1.00$) | Quantifies evidence completeness for clinical risk stratification. |
| **`inconsistency_severity`** | `score` | Continuous index ($0.00 \dots 1.00$) | Measures clinical danger of a contradiction (e.g. drug allergy conflict vs minor documentation typo). |

---

## 📊 Empirical Evaluation & Research Improvements

During development, we evaluated the original baseline checkpoint against our fine-tuned CEIS model. The baseline suffered from **severe multi-question semantic head failure**, which our fine-tuning resolved:

### 1. Head-to-Head Qualitative Evidence Comparison

| Case | Evidence & Claim | Baseline Behavior | CEIS (Fine-Tuned) Behavior | Research Significance |
|---|---|---|---|---|
| **Case 1: Definitive Corroboration** | **Claim:** *Patient diagnosed with pulmonary embolism.*<br>**Evidence:** *CT pulmonary angiogram demonstrated acute filling defect consistent with pulmonary embolism.* | Predicted `SUPPORTED`, but predicted `needs_more_evidence = 0.989` (98.9% chance more evidence is needed!) ❌ | Correctly predicted `SUPPORTED` with `needs_more_evidence = 0.346` (No more evidence required). ✅ | **Resolves head confusion**: CEIS aligns evidence sufficiency with diagnostic proof. |
| **Case 2: Explicit Clinical Rule-Out** | **Claim:** *Patient has acute kidney injury.*<br>**Evidence:** *Serum creatinine was stable at 0.7 mg/dL with robust urine output of 150 mL/hr, ruling out acute kidney injury.* | Predicted `CONTRADICTED`, but predicted `explicitly_mentioned = 0.014` (claims note never mentioned AKI!) ❌ | Correctly predicted `CONTRADICTED` with `explicitly_mentioned = 0.800` (Explicitly addressed). ✅ | **Proper topic awareness**: CEIS recognizes rule-outs as explicit clinical discussion. |

### 2. Multi-Task Loss Convergence

Training was conducted with hardware acceleration (`bfloat16`) on an **NVIDIA RTX 6000 Ada Generation (48 GB VRAM)**:
- **`verdict` loss**: Dropped from `0.4555` → **`0.2598`** (**-43% reduction**)
- **`explicitly_mentioned` loss**: Dropped from `1.4952` → **`0.5183`** (**-65% reduction**)
- **`needs_more_evidence` loss**: Dropped from `1.0864` → **`0.6412`** (**-41% reduction**)
- **Temperature calibration**: Calibrated on 2,406 records to $T = 1.1911$, reducing negative log-likelihood from `0.4791` to `0.4696` with mean confidence of `83.66%`.

---

## 📂 Project Architecture

```
ceis/
├── config/                  # Declarative configurations (YAML)
│   ├── train_config.yaml    # Fine-tuning hyperparameters, multi-task weights
│   └── eval_config.yaml     # Benchmark thresholds, temperature scaling
├── data/                    # Dataset acquisition & processing
│   ├── download.py          # MedNLI & PubMedQA automated downloaders
│   ├── clean.py             # Normalization, deduplication, clinical sanitization
│   ├── split.py             # Leakage-free group-aware splits across MIMIC-III IDs
│   └── synthetic_demos.json # 30 synthetic de-identified clinical test cases
├── training/                # LAYA multi-task training engine
│   ├── schema.py            # Typed decision questions (choice, noul, score)
│   ├── trainer.py           # Custom multi-task trainer with AMP bfloat16
│   ├── multitask.py         # Multi-task loss & silver labeling generator
│   └── hyperopt.py          # Optuna hyperparameter optimization
├── evaluation/              # Clinical validation suite
│   ├── benchmark.py         # 3-system comparative benchmarking
│   ├── calibration.py       # Temperature scaling, ECE & Brier score
│   ├── ood_eval.py          # Out-of-domain evaluation on PubMedQA
│   └── report.py            # Automated research paper table generator
├── inference/               # Production inference pipeline
│   ├── pipeline.py          # ClinicalEvidencePipeline (CEISPipeline)
│   ├── highlighter.py       # Span extraction and HTML evidence markup
│   ├── chunker.py           # Long clinical note sliding-window chunker
│   ├── contradiction.py     # Pairwise note-vs-note contradiction scanner
│   └── schemas.py           # Dataclass definitions (CEISResult, etc.)
├── app/                     # Full-stack user interface
│   ├── api.py               # High-performance FastAPI REST backend
│   ├── schemas.py           # Pydantic request/response validation
│   └── frontend/
│       ├── streamlit_app.py # Interactive multi-tab clinical dashboard
│       └── styles.css       # Clean healthcare typography styling
├── notebooks/               # Research & exploration notebooks
│   ├── 01_model_analysis.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_demo.ipynb
├── publish.py               # Hugging Face Model Card & upload script
├── main.py                  # Unified CLI orchestrator
├── Makefile                 # Developer convenience commands
└── requirements.txt         # Pinned production dependencies
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/reddyshailesh1/Ceis.git
cd Ceis
pip install -r requirements.txt
```

### 2. Run Inference from Python
```python
from inference.pipeline import CEISPipeline

# Initializes with local checkpoint or Hugging Face repo
pipeline = CEISPipeline(model_path="reddyshailesh1/ceis", device="cuda:0")

result = pipeline.verify(
    claim="Patient has severe acute pancreatitis.",
    evidence="Serum lipase level was markedly elevated at 1450 U/L and CT abdomen showed diffuse peripancreatic fluid collections and inflammatory stranding consistent with acute pancreatitis."
)

print("Verdict:                 ", result.verdict)
print("Confidence:              ", f"{result.confidence * 100:.1f}%")
print("Explicitly Mentioned:    ", result.explicitly_mentioned)
print("Needs More Evidence:     ", result.needs_more_evidence)
print("Evidence Sufficiency:    ", f"{result.evidence_sufficiency:.2f}")
print("Clinician Review Flag:   ", result.human_review_recommended)
print("Reasoning:               ", result.reasoning)
print("Supporting Spans:        ", result.supporting_spans)
```

### 3. Launch the Web Application

The interactive web dashboard includes:
- **Clinical Claim Verification**: Real-time multi-question decision cards.
- **Passage Highlighting**: Color-coded supporting (green) and contradicting (red) spans.
- **Contradiction Matrix Scanner**: Compares Admission Notes vs Discharge Summaries for discrepancies.
- **Evaluation Dashboard**: Visualizes calibration curves and Brier scores.
- **Audit Export**: One-click download of clinical reasoning audit trails in JSON or CSV.

```bash
# Launch the Streamlit dashboard
streamlit run app/frontend/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🌐 FastAPI REST Endpoints

Start the high-throughput REST API:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at **`http://localhost:8000/docs`**.

### Verification Request
```bash
curl -X POST "http://localhost:8000/verify" \
     -H "Content-Type: application/json" \
     -d '{
       "claim": "Patient has acute myocardial infarction.",
       "evidence": "ECG revealed ST elevation in leads V1-V4 with elevated troponin T of 1.8 ng/mL."
     }'
```

### Response
```json
{
  "verdict": "supported",
  "confidence": 0.842,
  "probabilities": {
    "supported": 0.842,
    "unsupported": 0.091,
    "contradicted": 0.067
  },
  "explicitly_mentioned": true,
  "needs_more_evidence": false,
  "evidence_sufficiency": 0.90,
  "inconsistency_severity": 0.0,
  "human_review_recommended": false,
  "reasoning": "The clinical evidence explicitly mentions the evaluated assertion. Classification decision: 'SUPPORTED' with 84.2% calibrated confidence.",
  "supporting_spans": [
    "ECG revealed ST elevation in leads V1-V4 with elevated troponin T of 1.8 ng/mL."
  ],
  "contradicting_spans": []
}
```

---

## 🛠️ CLI Commands & Makefile

The project includes a unified CLI (`main.py`) and standard `Makefile`:

| Task | Command | Description |
|---|---|---|
| **Data Acquisition** | `python main.py data download` | Fetches MedNLI and PubMedQA from Hugging Face Hub |
| **Data Processing** | `python main.py data clean && python main.py data split` | Deduplicates and performs leakage-free group-aware splitting |
| **Quick Test Run** | `python main.py train --limit 200 --epochs 1 --out runs/quick` | Verifies forward/backward pass in under 60 seconds |
| **Full Fine-Tuning** | `python main.py train --config config/train_config.yaml` | Multi-task training across full MedNLI dataset |
| **Model Benchmarking** | `python main.py eval benchmark` | Evaluates Base LAYA, MedNLI Baseline, and CEIS |
| **Start UI** | `python main.py ui` | Launches Streamlit frontend |
| **Start API** | `python main.py serve --port 8000` | Launches FastAPI backend |
| **Publish Checkpoint** | `python publish.py --model-dir runs/ceis_mednli_v1 --repo reddyshailesh1/ceis --public` | Uploads model weights and card to Hugging Face Hub |

---

## 📚 Citations & Acknowledgments

- **LAYA Engine**: [`convaiinnovations/laya`](https://github.com/NandhaKishorM/laya)
- **MedNLI Dataset**: Romanov & Shivade (2018), *Lessons from Natural Language Inference in the Clinical Domain*, via [`presencesw/mednli`](https://huggingface.co/datasets/presencesw/mednli)
- **PubMedQA Dataset**: Jin et al. (2019), *PubMedQA: A Dataset for Biomedical Research Question Answering*, via [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA)
- **Hugging Face Model Checkpoint**: [`reddyshailesh1/ceis`](https://huggingface.co/reddyshailesh1/ceis)

---

## 👤 Author
**Shailesh Reddy** ([@reddyshailesh1](https://github.com/reddyshailesh1))  
Project: Advanced Clinical Evidence Intelligence System (CEIS)
