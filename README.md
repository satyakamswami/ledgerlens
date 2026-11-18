# LedgerLens

<p align="center">
  <b>Production-grade Accounts Payable (AP) audit intelligence engine combining deterministic three-way matching controls, unsupervised anomaly detection, and bounded LLM tool-calling.</b>
</p>

<p align="center">
  <a href="https://ledgerlens-q0ug.onrender.com/"><b>View Live Deployment</b></a>
</p>

---

## 🏗️ System Architecture

```text
[ Raw Invoice / PO CSV Ingestion ]
                 │
                 ▼
      [ AuditEngine (Pandas / NumPy) ]
         ├── Deterministic Controls (Duplicate ID, Unit Price, Qty, Amount Delta)
         └── Unsupervised ML (IsolationForest on Scaled Behavioral Feature Vectors)
                 │
                 ▼
      [ Structured Findings & Evidence DataFrame ]
                 │
                 ├──► Streamlit Dashboard (Interactive UI, Metrics, CSV Export)
                 └──► FinancialAnalyst (Groq LLM API + Bounded JSON Tool-Calling)
```

---

## 🛠️ Technology Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **Frontend & UI** | Streamlit (Reactive component state, layout orchestration, data grid rendering) |
| **Data Processing** | Python, Pandas, NumPy (Vectorized transformations, type coercion, exposure aggregation) |
| **Machine Learning** | Scikit-Learn (`IsolationForest`, `StandardScaler`, evaluation metrics precision/recall/F1) |
| **AI / LLM Layer** | Requests, Groq API (OpenAI-compatible chat completion endpoints with strict tool-binding schemas) |
| **Infrastructure** | Render (Blueprint deployment via `render.yaml`, environment variable secret injection) |

---

## 📂 Project Directory Structure

```text
ledgerlens/
│
├── .gitignore          # Excludes virtual environments, cache, and local secrets
├── .env                # Local environment variables (API keys, model configuration)
├── render.yaml         # Render cloud blueprint specification
├── requirements.txt    # Pinned production dependencies
├── app.py              # Streamlit dashboard entrypoint and state orchestration
├── audit_engine.py     # Deterministic audit controls and Isolation Forest pipeline
├── llm_engine.py       # Groq-compatible LLM agent with bounded tool dispatchers
└── README.md           # Repository documentation
```

---

## 📊 Input CSV Schema Specification

To execute audits successfully, ingested CSV datasets must conform strictly to the following schema:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `invoice_id` | String | Unique invoice reference number issued by vendor |
| `vendor` | String | Commercial entity or supplier name |
| `po_id` | String | Approved Purchase Order identifier |
| `invoice_date` | Date (`YYYY-MM-DD`) | Issuance date of the vendor invoice |
| `sku` | String | Stock-keeping unit or service item code |
| `quantity` | Numeric | Units billed on the invoice line item |
| `unit_price` | Numeric | Billed unit price from the vendor invoice |
| `po_unit_price` | Numeric | Contractual baseline unit price on the Purchase Order |
| `po_quantity` | Numeric | Authorized quantity on the Purchase Order |
| `paid_amount` | Numeric | Total capital disbursed by the company |
| `expected_amount` | Numeric | Calculated baseline expected disbursement |

---

## 🚀 Local Installation & Development

### 1. Clone the Repository
```bash
git clone https://github.com/satyakamswami/ledgerlens.git
cd ledgerlens
```

### 2. Initialize Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Local Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### 4. Execute the Application
```bash
streamlit run app.py
```

---

## ⚙️ Core Engine Mechanics

1. **Deterministic Rule Engine (`audit_engine.py`)**: Executes vectorized checks for duplicate vendor invoice references, unit price overages against PO baselines, quantity inflation, and disbursement mismatches. Bounding is applied via `np.maximum.reduce` to prevent double-counting overlapping exception deltas.
2. **Behavioral Anomaly Detection**: Constructs standardized log-scale variance features (`unit_price_variance`, `quantity_variance`, `payment_variance`, `invoice_value_log`) and fits an `IsolationForest` to surface outlier transactions requiring compliance review.
3. **Grounded AI Analysis (`llm_engine.py`)**: Implements explicit schema-bound tool dispatchers (`vendor_exposure`, `invoice_evidence`, `risk_summary`). Forces the LLM to retrieve verifiable context via tool calls rather than hallucinating calculations across raw text.
