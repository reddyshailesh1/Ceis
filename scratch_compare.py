import json
import torch
import laya
from training.schema import CEISSchema

print("=== Quantitative Head-to-Head Evaluation ===")
print("Loading Baseline: pavanmantha/mednli_laya...")
base_agent = laya.load("pavanmantha/mednli_laya", device="cuda:0")

from pathlib import Path
ceis_dir = Path(__file__).resolve().parent / "runs" / "ceis_mednli_v1"
print(f"Loading CEIS Model from local path: {ceis_dir}...")
ceis_agent = laya.load(str(ceis_dir), device="cuda:0")

questions = {
    "verdict": CEISSchema.question(0),
    "explicitly_mentioned": CEISSchema.yesno_question(CEISSchema.QID_MENTIONED),
    "needs_more_evidence": CEISSchema.yesno_question(CEISSchema.QID_NEEDS_MORE),
}

test_cases = [
    {
        "name": "Corroborated Clinical Evidence",
        "claim": "Patient was diagnosed with pulmonary embolism.",
        "evidence": "CT pulmonary angiogram demonstrated acute filling defect in the right main pulmonary artery consistent with pulmonary embolism.",
        "gold": "supported"
    },
    {
        "name": "Direct Clinical Contradiction",
        "claim": "Patient has acute kidney injury.",
        "evidence": "Serum creatinine was stable at 0.7 mg/dL with robust urine output of 150 mL/hr, ruling out acute kidney injury.",
        "gold": "contradicted"
    },
    {
        "name": "Completely Irrelevant Evidence (Mismatch)",
        "claim": "Patient has severe hypertension requiring IV labetalol.",
        "evidence": "Patient underwent uncomplicated right knee arthroscopy under spinal anesthesia and was discharged home on crutches.",
        "gold": "unsupported"
    },
    {
        "name": "Insufficient / Ambiguous Clinical Evidence",
        "claim": "Patient had an ST-elevation myocardial infarction.",
        "evidence": "Patient complained of non-pleuritic chest heaviness after exertion; 12-lead ECG showed non-specific T-wave flattening in lead III; cardiac troponin I is pending.",
        "gold": "unsupported"
    }
]

for tc in test_cases:
    state = CEISSchema.state(tc["claim"], tc["evidence"])
    res_base = base_agent.predict(state, questions)
    res_ceis = ceis_agent.predict(state, questions)
    
    print("\n" + "="*50)
    print(f"CASE: {tc['name']}")
    print("Claim:   ", tc["claim"])
    print("Evidence:", tc["evidence"])
    print("-" * 50)
    
    b_v = res_base["answers"]["verdict"]["choice"]
    b_p = res_base["answers"]["verdict"]["probabilities"][b_v]
    b_ment = res_base["answers"]["explicitly_mentioned"]["noul"]
    b_need = res_base["answers"]["needs_more_evidence"]["noul"]
    
    c_v = res_ceis["answers"]["verdict"]["choice"]
    c_p = res_ceis["answers"]["verdict"]["probabilities"][c_v]
    c_ment = res_ceis["answers"]["explicitly_mentioned"]["noul"]
    c_need = res_ceis["answers"]["needs_more_evidence"]["noul"]
    
    print(f"BASELINE: Verdict={b_v.upper():12s} (p={b_p:.3f}) | Mentioned={b_ment:.3f} | NeedsMore={b_need:.3f}")
    print(f"CEIS:     Verdict={c_v.upper():12s} (p={c_p:.3f}) | Mentioned={c_ment:.3f} | NeedsMore={c_need:.3f}")
