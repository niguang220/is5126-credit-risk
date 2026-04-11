"""
IS5126 Credit Risk Demo — Streamlit App

Usage:
    streamlit run demo/app.py

Three tabs:
    1. Overview      — research background, pipeline diagram, key results
    2. Case Studies  — the two case studies from the paper
    3. Interactive   — pick any sample_3000 case, run the full pipeline
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import textwrap
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))
from demo.config import (
    SAMPLE_3000_PATH, MODEL_RESULTS_PATH, TABLE1_PATH, TABLE2_PATH, TABLE3_PATH,
    FIG_PART2_ROC_BERT, FIG_PART2_ROC_COMP, FIG_PART2_METRICS,
    FIG_PART2_CONSISTENCY, FIG_PART2_BIAS, FIG_PART3_RESULTS,
    THRESH_APPROVE, THRESH_DECLINE,
    CASE_STUDY_1_IDX, CASE_STUDY_2_IDX,
    COLORS,
)
from demo.pipeline import load_artifacts, score as run_pipeline, compute_shap

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IS5126 Credit Risk Demo",
    page_icon="💳",
    layout="wide",
)
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: flex-start !important;
    gap: 18px;
    background: #f8fafc;
    padding: 14px 18px 0 18px;
    border-radius: 16px 16px 0 0;
    border-bottom: 1px solid #e5e7eb;
    width: 100%;
    margin: 0;
}

/* each tab */
.stTabs [data-baseweb="tab"] {
    height: 58px;
    white-space: nowrap;
    background: transparent;
    border-radius: 14px 14px 0 0;
    color: #374151;
    font-size: 1.18rem;
    font-weight: 700;
    padding: 0 28px;
    border-bottom: 4px solid transparent;
}

/* hover */
.stTabs [data-baseweb="tab"]:hover {
    background: #f3f4f6;
    color: #111827;
}

/* selected tab */
.stTabs [aria-selected="true"] {
    background: #fff1f2 !important;
    color: #be123c !important;
    border-bottom: 4px solid #ef4444 !important;
}

/* hide default highlight */
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts (cached) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading models and data…")
def get_artifacts():
    return load_artifacts()

@st.cache_data(show_spinner="Loading sample data…")
def get_sample():
    df = pd.read_parquet(SAMPLE_3000_PATH)
    with open(MODEL_RESULTS_PATH) as f:
        model_results = json.load(f)
    return df, model_results

model, model_baseline, woe_maps, numeric_medians, qwen_cache, ood_bounds, model_feature_list = get_artifacts()
sample_df, model_results = get_sample()

# ── Helpers ───────────────────────────────────────────────────

def decision_badge(decision: str) -> str:
    color = {"Approve": COLORS["approve"], "Manual Review": COLORS["review"], "Decline": COLORS["decline"]}
    return f'<span style="background:{color[decision]};color:white;padding:4px 12px;border-radius:12px;font-weight:bold">{decision}</span>'

def score_gauge(ml_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ml_score,
        number={"valueformat": ".3f", "font": {"size": 36}},
        title={"text": "ML Risk Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": "#2E86AB"},
            "steps": [
                {"range": [0, THRESH_APPROVE], "color": "#d4edda"},
                {"range": [THRESH_APPROVE, THRESH_DECLINE], "color": "#fff3cd"},
                {"range": [THRESH_DECLINE, 1], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": ml_score,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
    return fig

def shap_waterfall(shap_items: list[dict]) -> go.Figure:
    feats  = [x["feature"] for x in reversed(shap_items)]
    values = [x["shap"]    for x in reversed(shap_items)]
    colors = [COLORS["decline"] if v > 0 else COLORS["approve"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=feats, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top SHAP Feature Impacts",
        xaxis_title="Impact on Risk Score",
        height=350,
        margin=dict(t=40, b=10, l=10, r=60),
        xaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1),
    )
    return fig


# ── TAB LAYOUT ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Overview", "Case Studies", "Interactive Demo"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.title("LLM-Enhanced Credit Risk Assessment")
    st.caption("IS5126 Group 7 · LendingClub Dataset · NUS 2026")

    st.markdown("""
    > **Research question:** Does LLM-based semantic analysis of loan descriptions
    > improve credit risk model accuracy — and does a stronger LLM produce better features?
    """)

    st.subheader("Pipeline Architecture")

    st.code(
        """Loan Application
    │
    ├─ Structured Features ──────────────────────► XGBoost ──► ML Risk Score
    │   (FICO, DTI, income, …)                                         │
    │                                                                  ▼
    └─ desc text (if available) ──► Qwen3-Max ──► qwen_score ──► Final Score + Reasoning
                                            ↑
                                LLM as Preprocessor
                                (not a co-decision-maker)""",
        language="text"
    )

    # Decision thresholds
    st.subheader("Decision Thresholds")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Approve",       f"Risk Score < {THRESH_APPROVE}",  "Actual default rate < 6.6%")
    c2.metric("🔶 Manual Review", f"{THRESH_APPROVE} – {THRESH_DECLINE}", "~25% of cases")
    c3.metric("❌ Decline",       f"Risk Score > {THRESH_DECLINE}",  "KS-optimal threshold (KS=0.318)")

    st.divider()

    # Part 1 results
    st.subheader("Part 1 — Traditional ML Baseline (Full Dataset, 1.3M loans)")
    part1_data = {
        "Model": ["LR — Exp A (w/ grade)", "XGB — Exp A (w/ grade)", "LGB — Exp A (w/ grade)",
                  "LR — Exp B (no grade)", "XGB — Exp B (no grade)", "LGB — Exp B (no grade)"],
        "AUC":  [0.7047, 0.7221, 0.6906, 0.6973, 0.7189, 0.6719],
        "KS":   [0.300,  0.324,  0.276,  0.288,  0.318,  0.248],
        "Gini": [0.410,  0.444,  0.381,  0.395,  0.438,  0.344],
    }
    p1_df = pd.DataFrame(part1_data).set_index("Model")
    num_cols_p1 = p1_df.select_dtypes(include="number").columns
    st.dataframe(
        p1_df.style.highlight_max(axis=0, color="#d4edda", subset=num_cols_p1).format("{:.4f}", subset=num_cols_p1),
        use_container_width=True,
    )
    st.caption("Exp A = includes grade/subgrade/int_rate. Exp B = excludes grade features (used as Part 2 baseline).")

    # Part 2 results
    st.subheader("Part 2 — LLM Semantic Feature Experiments")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Table 1 — BERT Effect (200K desc subset)**")
        t1 = pd.read_csv(TABLE1_PATH).set_index("Model")
        num_cols1 = t1.select_dtypes(include="number").columns
        st.dataframe(t1.style.format("{:.4f}", subset=num_cols1), use_container_width=True)

    with col_b:
        st.markdown("**Table 2 — LLM Strength Comparison (3K sample)**")
        t2 = pd.read_csv(TABLE2_PATH).set_index("Model")
        num_cols2 = t2.select_dtypes(include="number").columns
        st.dataframe(t2.style.format("{:.4f}", subset=num_cols2), use_container_width=True)

    st.markdown("""
    **Findings:**
    - Naive BERT: no significant improvement (ΔAUC = −0.002, p = 0.865)
    - Qwen3-Max: marginal improvement in **KS (+0.050)** and **BACC (+0.026)**, AUC slightly lower (−0.011)
    - Qwen consistency: std ≈ 0 across 5 runs → **near-perfect reproducibility**
    - Faithfulness: **88.2%** of reasoning directions match the numerical score
    """)

    if FIG_PART2_METRICS.exists():
        st.image(str(FIG_PART2_METRICS), caption="Part 2: Multi-metric comparison across pipelines")

    st.divider()

    # Part 3 results
    st.subheader("Part 3 — LLM Correction Experiment (200 samples, no desc)")
    t3 = pd.read_csv(TABLE3_PATH) if TABLE3_PATH.exists() else None
    if t3 is not None:
        st.dataframe(t3, use_container_width=True)
    else:
        correction_data = {
            "Condition": ["ML only (all 200)", "ML + LLM correction (all 200)",
                          "ML only (borderline 100)", "ML + LLM (borderline 100)",
                          "ML only (confident 100)"],
            "Accuracy": [0.645, 0.620, 0.600, 0.550, 0.690],
            "BACC":     [0.645, 0.620, 0.600, 0.550, 0.690],
        }
        st.dataframe(pd.DataFrame(correction_data).set_index("Condition"), use_container_width=True)

    st.error(
        "**Key finding:** LLM correction on borderline cases **reduced accuracy by 5pp** "
        "(0.600 → 0.550). Without free text, LLM has no information advantage over ML. "
        "This confirms LLM's value is in **text semantic extraction**, not structural reasoning.",
        icon="⚠️"
    )

    if FIG_PART3_RESULTS.exists():
        st.image(str(FIG_PART3_RESULTS), caption="Part 3: Correction experiment results")

    st.divider()
    st.subheader("MAS FEAT Compliance Angle")
    st.markdown("""
    | FEAT Principle | How this system addresses it |
    |---|---|
    | **Fairness** | SHAP values reveal which features drive each decision |
    | **Ethics** | Bounded LLM role — not a decision-maker, a signal extractor |
    | **Accountability** | Every prediction has a traceable feature attribution |
    | **Transparency** | Qwen `reasoning` provides natural language explanation per case |

    > LLM's strongest contribution under MAS FEAT is **explainability** —
    > providing readable, auditable reasoning that black-box XGBoost cannot.
    """)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — CASE STUDIES
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <style>
    .case-note {
        font-size: 1.02rem;
        color: #4b5563;
        margin-bottom: 1.1rem;
    }

    .case-card {
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        height: 100%;
    }

    .case-banner-good {
        background: #ecfdf3;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 0.75rem 0.95rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .case-banner-bad {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 0.75rem 0.95rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .case-section-title {
        font-size: 1rem;
        font-weight: 700;
        margin: 1rem 0 0.45rem 0;
        color: #111827;
    }

    .case-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.45rem 1rem;
        margin-bottom: 0.9rem;
    }

    .case-metric-item {
        font-size: 0.98rem;
        line-height: 1.5;
        color: #374151;
    }

    .case-box-blue {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        min-height: 160px;
        max-height: 190px;
        overflow-y: auto;
        color: #1d4ed8;
        line-height: 1.7;
    }
    .reasoning-wrap {
        min-height: 305px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .risk-wrap {
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .risk-list {
        margin: 0.4rem 0 0 0.2rem;
        padding-left: 1.3rem;
        min-height: 90px;
        color: #374151;
        line-height: 1.8;
    }

    .case-box-yellow {
        background: #fffbea;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        min-height: 160px;
        max-height: 190px;
        overflow-y: auto;
        color: #92400e;
        line-height: 1.7;
    }

    .case-box-gray {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        min-height: 170px;
        color: #374151;
        line-height: 1.75;
    }

    .score-line {
        margin: 0.85rem 0 0.25rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
    }

    .score-pill {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #d1d5db;
        background: #f3f4f6;
        color: #374151;
        margin-left: 0.25rem;
    }

    .risk-list {
        margin: 0.4rem 0 0 0.2rem;
        padding-left: 1.3rem;
        min-height: 105px;
        color: #374151;
        line-height: 1.8;
    }

    .summary-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        min-height: 165px;
    }

    .summary-label {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 0.45rem;
    }

    .summary-main {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }

    .summary-tag {
        display: inline-block;
        margin-top: 0.7rem;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: #ecfdf3;
        color: #166534;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .unified-box {
        margin-top: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: #fafafa;
        color: #374151;
        font-size: 1rem;
        line-height: 1.75;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Case Studies — When Does LLM Add Value?")
    st.markdown(
        '<div class="case-note">Two real loans from the LendingClub dataset under the same pipeline, '
        'but with very different outcomes. The contrast highlights the role of <code>desc</code> quality in shaping LLM signal.</div>',
        unsafe_allow_html=True
    )

    cs1_raw = sample_df.loc[CASE_STUDY_1_IDX]
    cs2_raw = sample_df.loc[CASE_STUDY_2_IDX]
    cs1_qwen = qwen_cache.get(str(CASE_STUDY_1_IDX), {})
    cs2_qwen = qwen_cache.get(str(CASE_STUDY_2_IDX), {})

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="case-card">', unsafe_allow_html=True)
        st.markdown("### Case A — Qwen Catches What ML Misses ✅")
        st.markdown(
            '<div class="case-banner-good">Qwen effective: desc reveals semantic risk signals beyond structured features</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="case-section-title">Loan Profile</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="case-metrics">
            <div class="case-metric-item"><b>Loan Amount:</b> ${int(cs1_raw['loan_amnt']):,}</div>
            <div class="case-metric-item"><b>Annual Income:</b> ${int(cs1_raw['annual_inc']):,}</div>
            <div class="case-metric-item"><b>DTI:</b> {cs1_raw['dti']:.2f}</div>
            <div class="case-metric-item"><b>FICO Score:</b> {int(cs1_raw['fico_score'])}</div>
            <div class="case-metric-item"><b>Purpose:</b> {cs1_raw['purpose']}</div>
            <div class="case-metric-item"><b>True Outcome:</b> DEFAULT 🔴</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="case-section-title">Borrower Description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="case-box-blue">{str(cs1_raw["desc"])[:600]}</div>', unsafe_allow_html=True)

        if cs1_qwen and not cs1_qwen.get("parse_error"):
            qwen_score_1 = cs1_qwen.get("score", 0.0)
            qwen_level_1 = cs1_qwen.get("risk_level", "")
            st.markdown(
                f'<div class="score-line">Qwen Risk Score: <code>{qwen_score_1:.2f}</code><span class="score-pill">{qwen_level_1}</span></div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="case-section-title">Qwen Reasoning</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="case-box-yellow">{cs1_qwen.get("reasoning", "")}</div>',
                unsafe_allow_html=True
            )

            risk_items = "".join([f"<li>{f}</li>" for f in cs1_qwen.get("key_risk_factors", [])])
            st.markdown('<div class="case-section-title">Key Risk Factors Identified by Qwen</div>', unsafe_allow_html=True)
            st.markdown(f'<ul class="risk-list">{risk_items}</ul>', unsafe_allow_html=True)

        st.markdown('<div class="case-section-title">Interpretation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="case-box-gray">
        ML sees a relatively safe structured profile and would likely approve this case.
        Qwen reads the borrower description and flags additional semantic warning signs,
        including weak business planning and unclear fund prioritisation.
        The borrower ultimately defaulted.
        <div class="case-highlight-good">
        Key takeaway: semantic features provide real incremental value beyond structured data in complex cases.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="case-card">', unsafe_allow_html=True)
        st.markdown("### Case B — Qwen Fails: Low-Info Description ❌")
        st.markdown(
            '<div class="case-banner-bad">Qwen ineffective: low-information desc provides little actionable signal</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="case-section-title">Loan Profile</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="case-metrics">
            <div class="case-metric-item"><b>Loan Amount:</b> ${int(cs2_raw['loan_amnt']):,}</div>
            <div class="case-metric-item"><b>Annual Income:</b> ${int(cs2_raw['annual_inc']):,}</div>
            <div class="case-metric-item"><b>DTI:</b> {cs2_raw['dti']:.1f}</div>
            <div class="case-metric-item"><b>FICO Score:</b> {int(cs2_raw['fico_score'])}</div>
            <div class="case-metric-item"><b>Purpose:</b> {cs2_raw['purpose']}</div>
            <div class="case-metric-item"><b>True Outcome:</b> DEFAULT 🔴</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="case-section-title">Borrower Description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="case-box-blue">{str(cs2_raw["desc"])}</div>', unsafe_allow_html=True)

        if cs2_qwen and not cs2_qwen.get("parse_error"):
            qwen_score_2 = cs2_qwen.get("score", 0.0)
            qwen_level_2 = cs2_qwen.get("risk_level", "")
            st.markdown(
                f'<div class="score-line">Qwen Risk Score: <code>{qwen_score_2:.2f}</code><span class="score-pill">{qwen_level_2}</span></div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="case-section-title">Qwen Reasoning</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="case-box-yellow">{cs2_qwen.get("reasoning", "")}</div>',
                unsafe_allow_html=True
            )

            if cs2_qwen.get("key_risk_factors"):
                risk_items = "".join([f"<li>{f}</li>" for f in cs2_qwen.get("key_risk_factors", [])])
            else:
                risk_items = "<li>Limited informative risk cues in desc</li>"
            st.markdown('<div class="case-section-title">Key Risk Factors Identified by Qwen</div>', unsafe_allow_html=True)
            st.markdown(f'<ul class="risk-list">{risk_items}</ul>', unsafe_allow_html=True)

        st.markdown('<div class="case-section-title">Interpretation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="case-box-gray">
        The description provides only a weak positive signal related to job stability
        and offers little additional context on repayment capacity.
        Qwen therefore assigns a relatively low risk score, yet the borrower defaults.
        <div class="case-highlight-bad">
        Key takeaway: when desc quality is low, LLM-derived signals become much less effective.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("LLM Quality Evaluation Summary")

    s1, s2, s3 = st.columns(3, gap="large")

    with s1:
        st.markdown("""
        <div class="summary-card">
            <div class="summary-label">Consistency</div>
            <div class="summary-main">std ≈ 0.003</div>
            <div class="summary-tag">100% samples std &lt; 0.05</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="summary-card">
            <div class="summary-label">Faithfulness</div>
            <div class="summary-main">88.2%</div>
            <div class="summary-tag">reasoning direction matches score</div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="summary-card">
            <div class="summary-label">Bias (purpose)</div>
            <div class="summary-main">r = 0.235</div>
            <div class="summary-tag">limited by desc quality variation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="unified-box">
    <b>Unified interpretation:</b> Qwen demonstrates strong stability and reasonably aligned reasoning,
    while the performance ceiling is primarily constrained by input <code>desc</code> quality.
    In settings where borrowers provide richer and more structured narratives,
    stronger semantic gains are likely.
    </div>
    """, unsafe_allow_html=True)

    if FIG_PART2_CONSISTENCY.exists():
        st.image(str(FIG_PART2_CONSISTENCY), caption="Consistency: near-zero variation across five independent runs")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — INTERACTIVE DEMO
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.title("Interactive Pipeline Demo")
    st.markdown(
        "Select any loan from the 3,000-sample dataset to run the full 5-step pipeline."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        thresh_approve = st.slider(
            "Approve threshold",
            min_value=0.10, max_value=0.45, value=THRESH_APPROVE, step=0.01,
            help="Below this → Approve. Default 0.25: actual default rate < 6.6% in this zone."
        )
        thresh_decline = st.slider(
            "Decline threshold",
            min_value=0.40, max_value=0.90, value=THRESH_DECLINE, step=0.01,
            help="Above this → Decline. Default 0.52: KS-optimal threshold (max TPR-FPR separation)."
        )
        st.caption("Approve < 0.25: actual default rate 6.6%  |  Decline > 0.52: KS-optimal (KS=0.318)")
        st.divider()
        st.markdown("**Note:** Risk Score is an uncalibrated model output, not a true probability. Thresholds are set on the same scale.")

    # Case selector — prioritise featured case studies, then high-quality desc cases
    with_qwen = [
        idx for idx in sample_df.index
        if str(idx) in qwen_cache
        and not qwen_cache[str(idx)].get("parse_error")
        and len(str(sample_df.loc[idx].get("desc", "") or "")) > 100
    ]
    featured = [CASE_STUDY_1_IDX, CASE_STUDY_2_IDX]
    other_idxs = [i for i in with_qwen if i not in featured][:48]
    display_idxs = featured + other_idxs  # cap at ~50 for usability

    def idx_label(i):
        desc_preview = str(sample_df.loc[i].get("desc", ""))[:60].replace("\n", " ")
        label = f"Case {i}"
        if i == CASE_STUDY_1_IDX:
            label += " ⭐ Startup+debt (Case Study A)"
        elif i == CASE_STUDY_2_IDX:
            label += " ⭐ Template text (Case Study B)"
        return label, f"{label} — {desc_preview}…"

    options = {idx_label(i)[1]: i for i in display_idxs}
    selected_label = st.selectbox("Select a loan application:", list(options.keys()))
    selected_idx = options[selected_label]
    raw = sample_df.loc[selected_idx]

    st.divider()

    # Run pipeline
    with st.spinner("Running pipeline…"):
        result = run_pipeline(
            raw, selected_idx,
            model, woe_maps, numeric_medians, qwen_cache, ood_bounds,
            model_feature_list, model_baseline,
        )
        # Apply sidebar thresholds
        from demo.pipeline import _make_decision
        ml_score  = result["ml_score"]
        decision  = _make_decision(ml_score, thresh_approve, thresh_decline)

    # ── Layout ────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Step 1 — Pre-screen
        st.markdown("#### Step 1 — Pre-screen")
        dq = result["desc_quality"]
        dq_map = {
            "good":    ("✅ desc present and informative", "success"),
            "short":   ("⚠️ desc too short — limited semantic signal", "warning"),
            "missing": ("❌ No desc — ML-only mode", "error"),
        }
        msg, level = dq_map[dq]
        getattr(st, level)(msg)

        # Step 2 — OOD
        st.markdown("#### Step 2 — OOD Detection")
        ood = result["ood"]
        if ood["is_ood"]:
            st.warning(
                f"⚠️ Unusual Profile — {ood['n_ood']} / {len(ood['details'])} key features "
                f"fall outside training distribution. Prediction may be less reliable."
            )
        else:
            st.success(f"✅ Within training distribution ({ood['n_ood']} / {len(ood['details'])} features flagged)")

        with st.expander("OOD feature details"):
            rows = []
            for feat, d in ood["details"].items():
                rows.append({
                    "Feature": feat,
                    "Value": f"{d['value']:.2f}" if d["value"] is not None else "N/A",
                    "Training range": f"[{d['lo']:.2f}, {d['hi']:.2f}]",
                    "OOD": "⚠️" if d["ood"] else "✅",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True)

        # Step 3 — ML Scoring (before/after Qwen)
        st.markdown("#### Step 3 — ML Scoring")
        baseline_score = result.get("ml_baseline_score")
        if baseline_score is not None:
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Without Qwen** (95 structured features)")
                st.plotly_chart(score_gauge(baseline_score), use_container_width=True)
                base_dec = _make_decision(baseline_score, thresh_approve, thresh_decline)
                base_color = {"Approve": COLORS["approve"], "Manual Review": COLORS["review"], "Decline": COLORS["decline"]}[base_dec]
                st.markdown(f'<div style="text-align:center;color:{base_color};font-weight:bold">{base_dec}</div>', unsafe_allow_html=True)
            with g2:
                st.markdown("**With Qwen** (95 features + qwen_score)")
                st.plotly_chart(score_gauge(ml_score), use_container_width=True)
                dec_color = {"Approve": COLORS["approve"], "Manual Review": COLORS["review"], "Decline": COLORS["decline"]}[decision]
                st.markdown(f'<div style="text-align:center;color:{dec_color};font-weight:bold">{decision}</div>', unsafe_allow_html=True)
            delta = ml_score - baseline_score
            arrow = "↑" if delta > 0 else "↓"
            st.caption(f"qwen_score = {result['qwen']['score'] if result['qwen'] else 'N/A'}  |  Score delta: {arrow} {abs(delta):.3f}  |  Thresholds: Approve < {thresh_approve:.2f} | Decline > {thresh_decline:.2f}")
        else:
            st.plotly_chart(score_gauge(ml_score), use_container_width=True)
            st.caption(f"Approve < {thresh_approve:.2f}  |  Manual Review {thresh_approve:.2f}–{thresh_decline:.2f}  |  Decline > {thresh_decline:.2f}")

    with col_right:
        # Step 4 — Qwen
        st.markdown("#### Step 4 — Qwen Semantic Analysis")
        qwen = result["qwen"]
        if qwen:
            q_score = qwen.get("score", None)
            q_level = qwen.get("risk_level", "")
            color_map = {"LOW": COLORS["approve"], "MEDIUM": COLORS["review"], "HIGH": COLORS["decline"]}
            badge_color = color_map.get(q_level, COLORS["neutral"])

            qc1, qc2 = st.columns(2)
            qc1.metric("Qwen Score", f"{q_score:.3f}" if q_score is not None else "N/A")
            qc2.markdown(
                f'<br><span style="background:{badge_color};color:white;padding:4px 12px;'
                f'border-radius:8px">{q_level}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("**Reasoning:**")
            st.info(qwen.get("reasoning", ""))

            if qwen.get("key_risk_factors"):
                st.markdown("**Risk factors:**")
                for f in qwen["key_risk_factors"]:
                    st.markdown(f"- 🔴 {f}")
            if qwen.get("key_protective_factors"):
                st.markdown("**Protective factors:**")
                for f in qwen["key_protective_factors"]:
                    st.markdown(f"- 🟢 {f}")
        elif dq == "good":
            st.warning("Qwen result not in cache for this case (parse error).")
        else:
            st.info("No desc available — Qwen analysis skipped. Prediction based on ML only.")

        # Step 5 — Decision
        st.markdown("#### Step 5 — Final Decision")
        dec_color = {"Approve": COLORS["approve"], "Manual Review": COLORS["review"], "Decline": COLORS["decline"]}
        st.markdown(
            f'<div style="text-align:center;padding:20px;background:{dec_color[decision]};'
            f'color:white;border-radius:12px;font-size:28px;font-weight:bold">{decision}</div>',
            unsafe_allow_html=True,
        )
        true_label = int(raw.get("default", -1))
        if true_label in (0, 1):
            outcome = "DEFAULT" if true_label == 1 else "REPAID"
            outcome_color = COLORS["decline"] if true_label == 1 else COLORS["approve"]
            st.markdown(
                f'<div style="text-align:center;margin-top:8px;color:{outcome_color};font-weight:bold">'
                f'True outcome: {outcome}</div>',
                unsafe_allow_html=True,
            )

    # SHAP
    st.divider()
    st.markdown("#### SHAP Feature Attribution")
    with st.spinner("Computing SHAP values…"):
        shap_items = compute_shap(model, result["feature_df"], top_n=10)
    st.plotly_chart(shap_waterfall(shap_items), use_container_width=True)
    st.caption(
        "Red bars increase risk score. Green bars decrease risk score. "
        "Values are SHAP contributions — not raw feature values."
    )

    # Raw feature summary
    with st.expander("Loan application details"):
        display_fields = [
            "loan_amnt", "annual_inc", "dti", "fico_score", "purpose",
            "home_ownership", "emp_length_num", "term", "installment",
            "revol_util", "inq_last_6mths", "delinq_2yrs",
        ]
        info = {f: raw.get(f) for f in display_fields if f in raw.index}
        info_df = pd.DataFrame.from_dict(info, orient="index", columns=["Value"])
        st.dataframe(info_df, use_container_width=True)

        if raw.get("desc"):
            st.markdown("**Full desc text:**")
            st.text(str(raw["desc"]))
