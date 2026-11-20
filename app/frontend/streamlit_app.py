"""
CEIS Streamlit Frontend
=======================
Run with:
    streamlit run app/frontend/streamlit_app.py

Default backend: http://localhost:8000
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CEIS — Clinical Evidence Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load custom CSS
# ---------------------------------------------------------------------------
_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "pavanmantha/mednli_laya"

# ---------------------------------------------------------------------------
# Sidebar — global config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    st.session_state.backend_url = st.text_input(
        "Backend URL", value=st.session_state.backend_url
    )

    # Try to fetch available models
    _model_options = [
        "pavanmantha/mednli_laya",
        "roberta-large-mnli",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    ]
    try:
        _r = requests.get(
            f"{st.session_state.backend_url}/api/v1/models", timeout=3
        )
        if _r.ok:
            _data = _r.json()
            _model_options = _data.get("available", _model_options)
    except Exception:
        pass

    st.session_state.selected_model = st.selectbox(
        "Active Model",
        options=_model_options,
        index=_model_options.index(st.session_state.selected_model)
        if st.session_state.selected_model in _model_options
        else 0,
    )

    st.divider()
    st.caption("CEIS v1.0 · Research only · Not for clinical use")

    # Health indicator
    try:
        _h = requests.get(
            f"{st.session_state.backend_url}/health", timeout=3
        )
        if _h.ok:
            _hdata = _h.json()
            _status = _hdata.get("status", "unknown")
            _color = "🟢" if _status == "ok" else "🟡"
            st.write(f"{_color} Backend: **{_status}** | Device: `{_hdata.get('device', '?')}`")
            if _hdata.get("loaded_models"):
                st.caption("Loaded: " + ", ".join(_hdata["loaded_models"]))
        else:
            st.write("🔴 Backend not reachable")
    except Exception:
        st.write("🔴 Backend not reachable")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _api(method: str, path: str, **kwargs) -> dict:
    url = f"{st.session_state.backend_url}{path}"
    resp = requests.request(method, url, timeout=120, **kwargs)
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def _verdict_badge(verdict: str) -> str:
    css_class = {
        "supported": "badge-supported",
        "unsupported": "badge-unsupported",
        "contradicted": "badge-contradicted",
    }.get(verdict.lower(), "badge-unsupported")
    return f'<span class="badge {css_class}">{verdict.upper()}</span>'


def _add_to_history(result: dict, claim: str):
    st.session_state.history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "claim": claim[:80] + ("…" if len(claim) > 80 else ""),
            "verdict": result.get("verdict", ""),
            "confidence": round(result.get("confidence", 0.0), 4),
            "review_recommended": result.get("human_review_recommended", False),
            "model": st.session_state.selected_model,
        }
    )


def _render_verify_result(result: dict):
    """Render a VerifyResponse dict as rich Streamlit widgets."""
    verdict = result.get("verdict", "unknown")
    confidence = result.get("confidence", 0.0)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Verdict:** {_verdict_badge(verdict)}", unsafe_allow_html=True)
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    with col2:
        probs = result.get("probabilities", {})
        if probs:
            fig = go.Figure(
                go.Bar(
                    x=list(probs.values()),
                    y=list(probs.keys()),
                    orientation="h",
                    marker_color=["#16a34a", "#d97706", "#dc2626"][: len(probs)],
                )
            )
            fig.update_layout(
                height=120,
                margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(range=[0, 1], title="Probability"),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"probs_{time.time()}")

    # Decision cards
    st.markdown("**Decision Outputs**")
    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    with dc1:
        v = "✅ Yes" if result.get("explicitly_mentioned") else "❌ No"
        st.metric("Explicitly Mentioned", v)
    with dc2:
        v = "✅ Yes" if result.get("needs_more_evidence") else "❌ No"
        st.metric("Needs More Evidence", v)
    with dc3:
        st.metric("Evidence Sufficiency", f"{result.get('evidence_sufficiency', 0):.2f}")
    with dc4:
        st.metric("Inconsistency Severity", f"{result.get('inconsistency_severity', 0):.2f}")
    with dc5:
        v = "⚠️ Yes" if result.get("human_review_recommended") else "✅ No"
        st.metric("Human Review", v)

    # Human review warning
    if result.get("human_review_recommended"):
        st.markdown(
            '<div class="review-warning">⚠️ Human expert review is recommended for this prediction.</div>',
            unsafe_allow_html=True,
        )

    # Highlighted evidence
    html = result.get("highlighted_evidence_html", "")
    if html:
        st.markdown("**Highlighted Evidence**")
        st.markdown(html, unsafe_allow_html=True)

    # Spans
    spans_col1, spans_col2 = st.columns(2)
    with spans_col1:
        sup = result.get("supporting_spans", [])
        if sup:
            st.success("**Supporting spans**")
            for s in sup:
                st.markdown(f'<span class="support-highlight">{s}</span>', unsafe_allow_html=True)
    with spans_col2:
        con = result.get("contradicting_spans", [])
        if con:
            st.error("**Contradicting spans**")
            for s in con:
                st.markdown(f'<span class="contradict-highlight">{s}</span>', unsafe_allow_html=True)

    # Reasoning
    with st.expander("💬 Reasoning"):
        st.write(result.get("reasoning", "No reasoning available."))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
(
    tab_verify,
    tab_scan,
    tab_sufficiency,
    tab_history,
    tab_eval,
    tab_about,
) = st.tabs(
    [
        "🔍 Claim Verifier",
        "🔄 Contradiction Scanner",
        "📊 Evidence Sufficiency",
        "📋 Decision History",
        "📈 Evaluation Dashboard",
        "ℹ️ About & Disclaimers",
    ]
)

# ===========================================================================
# Tab 1 — Claim Verifier
# ===========================================================================
with tab_verify:
    st.header("🔍 Claim Verifier")
    st.caption("Verify a single clinical claim against an evidence passage.")

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        claim_input = st.text_area(
            "Clinical Claim",
            height=120,
            placeholder="e.g., The patient's potassium level is within normal range.",
            key="v_claim",
        )
    with v_col2:
        evidence_input = st.text_area(
            "Evidence Passage",
            height=120,
            placeholder="e.g., Lab results show K+ = 4.1 mEq/L (normal 3.5–5.0).",
            key="v_evidence",
        )

    verify_btn = st.button("▶ Verify", type="primary", key="verify_btn")

    if verify_btn:
        if not claim_input.strip() or not evidence_input.strip():
            st.warning("Please provide both a claim and an evidence passage.")
        else:
            with st.spinner("Running inference…"):
                try:
                    result = _api(
                        "POST",
                        "/api/v1/verify",
                        json={
                            "claim": claim_input,
                            "evidence": evidence_input,
                            "model": st.session_state.selected_model,
                        },
                    )
                    _render_verify_result(result)
                    _add_to_history(result, claim_input)
                    st.success("Prediction added to Decision History.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

# ===========================================================================
# Tab 2 — Contradiction Scanner
# ===========================================================================
with tab_scan:
    st.header("🔄 Contradiction Scanner")
    st.caption(
        "Paste clinical statements (one per line). "
        "The system checks every ordered pair for contradictions."
    )

    statements_text = st.text_area(
        "Clinical Statements (one per line)",
        height=200,
        placeholder=(
            "Patient has type 2 diabetes.\n"
            "Patient was prescribed insulin for type 1 diabetes.\n"
            "Blood glucose is 90 mg/dL (normal)."
        ),
        key="scan_statements",
    )

    scan_btn = st.button("▶ Run Scan", type="primary", key="scan_btn")

    if scan_btn:
        stmts = [s.strip() for s in statements_text.splitlines() if s.strip()]
        if len(stmts) < 2:
            st.warning("Please enter at least 2 statements.")
        elif len(stmts) > 50:
            st.warning("Please enter at most 50 statements.")
        else:
            with st.spinner(f"Scanning {len(stmts)} statements ({len(stmts)**2 - len(stmts)} pairs)…"):
                try:
                    result = _api(
                        "POST",
                        "/api/v1/scan",
                        json={
                            "statements": stmts,
                            "model": st.session_state.selected_model,
                        },
                    )

                    st.info(result.get("summary", ""))

                    matrix = result.get("matrix", [])
                    if matrix:
                        # Encode as numeric for heatmap
                        label_map = {"self": 0, "supported": 1, "unsupported": 2, "contradicted": 3, "": 2}
                        color_scale = [
                            [0.0, "#cbd5e1"],   # self — slate
                            [0.33, "#86efac"],  # supported — green
                            [0.66, "#fde68a"],  # unsupported — amber
                            [1.0, "#fca5a5"],   # contradicted — red
                        ]
                        z = [[label_map.get(cell, 2) for cell in row] for row in matrix]
                        short_labels = [s[:30] + "…" if len(s) > 30 else s for s in stmts]

                        fig = go.Figure(
                            go.Heatmap(
                                z=z,
                                x=short_labels,
                                y=short_labels,
                                text=matrix,
                                texttemplate="%{text}",
                                colorscale=color_scale,
                                showscale=False,
                            )
                        )
                        fig.update_layout(
                            title="Pairwise Verdict Heatmap (row=claim, col=evidence)",
                            height=max(400, 60 * len(stmts)),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    flagged = result.get("flagged_pairs", [])
                    if flagged:
                        st.subheader(f"⚠️ {len(flagged)} Contradictory Pair(s) Found")
                        for fp in flagged:
                            with st.expander(
                                f"[{fp['claim_idx']}→{fp['evidence_idx']}] "
                                f"confidence={fp.get('confidence', 0):.1%} "
                                f"severity={fp.get('inconsistency_severity', 0):.2f}"
                            ):
                                st.markdown(
                                    f'<div class="flagged-pair">'
                                    f"<b>Claim:</b> {fp['claim']}<br><br>"
                                    f"<b>Evidence:</b> {fp['evidence']}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.success("No contradictions detected.")

                except Exception as exc:
                    st.error(f"Error: {exc}")

# ===========================================================================
# Tab 3 — Evidence Sufficiency
# ===========================================================================
with tab_sufficiency:
    st.header("📊 Evidence Sufficiency")
    st.caption(
        "Provide one evidence block and multiple claims. "
        "The chart shows how sufficient the evidence is per claim."
    )

    suf_evidence = st.text_area(
        "Evidence Block",
        height=150,
        placeholder="Paste the evidence passage here…",
        key="suf_evidence",
    )
    suf_claims_text = st.text_area(
        "Claims to Evaluate (one per line)",
        height=150,
        placeholder=(
            "Patient has hypertension.\n"
            "Patient is on metformin.\n"
            "Patient's renal function is normal."
        ),
        key="suf_claims",
    )

    suf_btn = st.button("▶ Analyse Sufficiency", type="primary", key="suf_btn")

    if suf_btn:
        suf_claims = [c.strip() for c in suf_claims_text.splitlines() if c.strip()]
        if not suf_evidence.strip():
            st.warning("Please provide an evidence block.")
        elif not suf_claims:
            st.warning("Please provide at least one claim.")
        else:
            with st.spinner("Running batch inference…"):
                try:
                    pairs = [{"claim": c, "evidence": suf_evidence} for c in suf_claims]
                    result = _api(
                        "POST",
                        "/api/v1/batch",
                        json={
                            "pairs": pairs,
                            "model": st.session_state.selected_model,
                        },
                    )
                    results = result.get("results", [])

                    rows = []
                    for claim, r in zip(suf_claims, results):
                        rows.append(
                            {
                                "Claim": claim[:60] + ("…" if len(claim) > 60 else ""),
                                "Verdict": r.get("verdict", ""),
                                "Sufficiency": r.get("evidence_sufficiency", 0.0),
                                "Confidence": r.get("confidence", 0.0),
                            }
                        )

                    df = pd.DataFrame(rows)

                    # Color-code bars by verdict
                    color_map = {
                        "supported": "#16a34a",
                        "unsupported": "#d97706",
                        "contradicted": "#dc2626",
                        "error": "#94a3b8",
                    }
                    df["color"] = df["Verdict"].map(color_map).fillna("#94a3b8")

                    fig = go.Figure()
                    for _, row in df.iterrows():
                        fig.add_bar(
                            x=[row["Claim"]],
                            y=[row["Sufficiency"]],
                            marker_color=row["color"],
                            name=row["Verdict"],
                            showlegend=False,
                        )
                    fig.update_layout(
                        title="Evidence Sufficiency per Claim",
                        yaxis=dict(range=[0, 1], title="Sufficiency Score"),
                        xaxis=dict(title="Claim"),
                        height=400,
                        barmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(df[["Claim", "Verdict", "Sufficiency", "Confidence"]], use_container_width=True)

                except Exception as exc:
                    st.error(f"Error: {exc}")

# ===========================================================================
# Tab 4 — Decision History
# ===========================================================================
with tab_history:
    st.header("📋 Decision History")
    history = st.session_state.history

    if not history:
        st.info("No predictions yet. Run a verification in the Claim Verifier tab.")
    else:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)

        # Downloads
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 4])
        with dl_col1:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇ Download CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="ceis_history.csv",
                mime="text/csv",
            )
        with dl_col2:
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps(history, indent=2).encode("utf-8"),
                file_name="ceis_history.json",
                mime="application/json",
            )
        with dl_col3:
            if st.button("🗑 Clear History"):
                st.session_state.history = []
                st.rerun()

# ===========================================================================
# Tab 5 — Evaluation Dashboard
# ===========================================================================
with tab_eval:
    st.header("📈 Evaluation Dashboard")
    st.caption(
        "Upload a `benchmark_results.json` produced by `evaluation/benchmark.py` "
        "to visualise model comparison metrics."
    )

    uploaded = st.file_uploader("Upload benchmark_results.json", type=["json"])

    if uploaded is not None:
        try:
            bench = json.load(uploaded)
        except Exception as exc:
            st.error(f"Could not parse JSON: {exc}")
            bench = None

        if bench:
            models_data = bench if isinstance(bench, list) else bench.get("models", [bench])

            # ---- Comparison table ----
            rows = []
            for m in models_data:
                rows.append(
                    {
                        "Model": m.get("model_id", "unknown"),
                        "Accuracy": m.get("accuracy", None),
                        "Macro F1": m.get("macro_f1", None),
                        "ECE": m.get("ece", None),
                        "Dataset": m.get("dataset", ""),
                    }
                )
            comp_df = pd.DataFrame(rows)
            st.subheader("Model Comparison")
            st.dataframe(comp_df, use_container_width=True)

            # ---- Accuracy bar chart ----
            if "Accuracy" in comp_df.columns and comp_df["Accuracy"].notna().any():
                fig_acc = px.bar(
                    comp_df,
                    x="Model",
                    y="Accuracy",
                    color="Model",
                    title="Accuracy by Model",
                    text_auto=".2%",
                )
                fig_acc.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_acc, use_container_width=True)

            # ---- Confusion matrices ----
            for m in models_data:
                cm = m.get("confusion_matrix")
                labels = m.get("labels", ["supported", "unsupported", "contradicted"])
                if cm:
                    st.subheader(f"Confusion Matrix — {m.get('model_id', 'Model')}")
                    fig_cm = ff.create_annotated_heatmap(
                        z=cm,
                        x=labels,
                        y=labels,
                        colorscale="Blues",
                    )
                    fig_cm.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        height=350,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

            # ---- Reliability diagrams ----
            for m in models_data:
                rel = m.get("reliability")
                if rel:
                    st.subheader(f"Reliability Diagram — {m.get('model_id', 'Model')}")
                    rel_df = pd.DataFrame(rel)
                    fig_rel = px.scatter(
                        rel_df,
                        x="confidence",
                        y="accuracy",
                        title="Reliability (Calibration) Diagram",
                        labels={"confidence": "Mean Confidence", "accuracy": "Fraction Correct"},
                    )
                    fig_rel.add_shape(
                        type="line",
                        x0=0, y0=0, x1=1, y1=1,
                        line=dict(color="gray", dash="dash"),
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)
    else:
        st.info(
            "No file uploaded yet. Run `python evaluation/benchmark.py` first to generate results."
        )

# ===========================================================================
# Tab 6 — About & Disclaimers
# ===========================================================================
with tab_about:
    st.header("ℹ️ About CEIS")

    st.markdown(
        """
## Clinical Evidence Intelligence System

CEIS is a **research prototype** that applies layered Natural Language Inference (NLI)
to assist clinicians in verifying whether clinical claims are supported, unsupported, or
contradicted by provided evidence passages.

### Architecture

```
Streamlit Frontend  ──HTTP──▶  FastAPI Backend  ──▶  CEISPipeline
                                                        │
                                               laya multi-question NLI
                                               (pavanmantha/mednli_laya)
                                                        │
                                               CEISResult (5 decisions)
```

The inference layer fires **5 questions** per (claim, evidence) pair:

| # | Question | Type |
|---|----------|------|
| 1 | Verdict | 3-way choice: supported / unsupported / contradicted |
| 2 | Explicitly mentioned? | Yes / No |
| 3 | Needs more evidence? | Yes / No |
| 4 | Evidence sufficiency | Score 0–1 |
| 5 | Inconsistency severity | Score 0–1 |

### Models

- **[pavanmantha/mednli_laya](https://huggingface.co/pavanmantha/mednli_laya)** — Fine-tuned on MedNLI with the CEIS multi-question schema (default)
- **roberta-large-mnli** — General-domain NLI baseline
- **microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext** — Biomedical BERT zero-shot baseline

### GitHub

> _Repository link will be added after publication._
        """
    )

    st.markdown(
        """
---
<div class="disclaimer-box">
⚠️ <strong>DISCLAIMER — RESEARCH USE ONLY</strong><br><br>
This tool is provided for <em>research and educational purposes only</em>. It is 
<strong>not validated for clinical use</strong>, is <strong>not a medical device</strong>, 
and should <strong>never be used to inform clinical decisions</strong> about real patients.<br><br>
Predictions may be incorrect. Always consult a qualified healthcare professional.
AI outputs in this system have not undergone regulatory review.
</div>
        """,
        unsafe_allow_html=True,
    )
