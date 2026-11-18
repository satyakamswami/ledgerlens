# LedgerLens — AP Leakage Detection & AI Audit Analyst

LedgerLens is a small, evidence-first accounts-payable audit prototype. It combines explainable transaction controls, an unsupervised anomaly signal, and an LLM analyst that can call bounded Python tools to inspect the audit results.

## Why it matters

Payment leakage can hide in duplicate invoices, invoice prices above purchase-order prices, quantities above approved quantities, and payments above expected amounts. LedgerLens turns these checks into traceable findings instead of asking an LLM to guess from raw financial data.

## Capabilities

- Validates invoice/PO data before analysis.
- Detects duplicate vendor invoice IDs, price variance, quantity variance, and paid-vs-expected mismatches.
- Uses Pandas/NumPy for preprocessing and feature engineering.
- Uses Scikit-learn Isolation Forest as a secondary anomaly-review signal.
- Assigns a transparent risk score and preserves invoice-level evidence.
- Supports zero-shot, one-shot, and few-shot analyst prompting.
- Uses LLM tool calling for vendor exposure, invoice evidence, and audit summaries.
- Exports findings to CSV.

## Stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, Groq API (Llama-compatible tool calling).

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The deterministic audit works without an API key. To enable the AI analyst, set:

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_key"

# macOS/Linux
export GROQ_API_KEY="your_key"
```

You can override the model with `GROQ_MODEL`.

## Input CSV schema

Required columns:

`invoice_id,vendor,po_id,invoice_date,sku,quantity,unit_price,po_unit_price,po_quantity,paid_amount,expected_amount`

Amounts are assumed to use one consistent currency within a file. The sample dataset is synthetic and intended only for demonstration.

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Render reads `render.yaml`; add your `GROQ_API_KEY` in the service environment.
4. Deploy and open the generated public URL.

## Engineering notes

- Rule-based exceptions are explainable controls; the Isolation Forest result is only a prioritization signal, not a fraud determination.
- Potential exposure is an estimate based on the supplied invoice/PO fields, not a confirmed recoverable amount.
- This demo does not connect to real ERP systems and should not be used to approve or block payments.
- For production use: add authentication, tenant isolation, audit logs, encrypted storage, role-based access, data retention controls, stronger schema/versioning, and a labelled evaluation dataset.

## Suggested evaluation

Create a labelled holdout dataset and report precision, recall, F1, false-positive rate, and review time saved. Do not report accuracy alone for imbalanced financial exception data.
