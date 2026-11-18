import os
import pandas as pd
import streamlit as st
from audit_engine import AuditEngine, REQUIRED_COLUMNS, demo_data
from llm_engine import FinancialAnalyst
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LedgerLens | AP Audit Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Precise minimalist CSS matching your aesthetic (pure black buttons, clean white cards, subtle borders, sleek layout)
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .main {
        background-color: #ffffff;
    }
    /* Metric Cards */
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #eaeaea;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .stMetric label {
        color: #666666 !important;
        font-weight: 500;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 600;
    }
    /* Headers */
    h1, h2, h3 {
        color: #000000;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        letter-spacing: -0.03em;
    }
    /* Sidebar */
    .stSidebar {
        background-color: #fafafa;
        border-right: 1px solid #eaeaea;
    }
    /* Primary Buttons (Stark Black) */
    .stButton>button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: opacity 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.85 !important;
    }
    /* Download Buttons */
    .stDownloadButton>button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stDownloadButton>button:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    /* Inputs */
    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    </style>
""", unsafe_allow_html=True)

st.title("LedgerLens")
st.caption("Accounts Payable Leakage Detection · Evidence-first AI audit assistant")

with st.sidebar:
    st.header("Audit workspace")
    uploaded = st.file_uploader("Upload invoice / PO CSV", type=["csv"])
    prompt_style = st.selectbox("Prompt strategy", ["zero-shot", "one-shot", "few-shot"])
    st.caption("Expected fields: " + ", ".join(REQUIRED_COLUMNS))
    st.divider()
    st.caption("Demo records are synthetic. Do not upload confidential financial data to a public demo.")

try:
    raw = pd.read_csv(uploaded) if uploaded else demo_data()
    engine = AuditEngine()
    findings, metrics, scored = engine.run(raw)
except Exception as exc:
    st.error(f"Could not process this file: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Invoices reviewed", f"{metrics['invoice_count']:,}")
c2.metric("Potential exposure", f"${metrics['exposure']:,.2f}")
c3.metric("Flagged records", f"{metrics['flagged_count']:,}")
c4.metric("High-risk vendors", f"{metrics['high_risk_vendors']:,}")
if "evaluation" in metrics:
    e1, e2, e3 = st.columns(3)
    e1.metric("Precision", metrics["evaluation"]["precision"])
    e2.metric("Recall", metrics["evaluation"]["recall"])
    e3.metric("F1 score", metrics["evaluation"]["f1"])

st.subheader("Audit findings")
if findings.empty:
    st.success("No rule-based exceptions were detected in this dataset.")
else:
    display_cols = ["invoice_id", "vendor", "po_id", "risk_level", "risk_score", "issue_types", "potential_overpayment"]
    st.dataframe(findings[display_cols], use_container_width=True, hide_index=True)

st.download_button(
    "Download audit findings (CSV)",
    findings.to_csv(index=False).encode("utf-8"),
    file_name="ledgerlens_audit_findings.csv",
    mime="text/csv",
)

with st.expander("How the detection engine works"):
    st.markdown(
        "- **Deterministic controls:** duplicate invoice IDs, unit-price variance, quantity over PO, and paid-vs-expected amount mismatch.\n"
        "- **Unsupervised signal:** Isolation Forest scores unusual transaction patterns; it is a review signal, not proof of fraud.\n"
        "- **Risk score:** combines explainable rule hits with the anomaly signal. Every flag retains its supporting evidence."
    )
    st.dataframe(scored.head(30), use_container_width=True, hide_index=True)

st.subheader("Ask the AI audit analyst")
question = st.text_input(
    "Ask about the current audit",
    placeholder="Which vendor has the largest potential overpayment, and why?"
)
if st.button("Analyse", type="primary", disabled=not question.strip()):
    analyst = FinancialAnalyst()
    context = {
        "metrics": metrics,
        "findings": findings.head(50).to_dict(orient="records"),
        "available_tools": ["vendor_exposure", "invoice_evidence", "risk_summary"],
    }
    with st.spinner("Reviewing the audit evidence..."):
        try:
            answer = analyst.answer(question, context, prompt_style)
            st.markdown(answer)
        except Exception as exc:
            st.error(str(exc))

st.caption("LedgerLens is a decision-support prototype. Findings require human validation before recovery or payment action.")
