import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

REQUIRED_COLUMNS = [
    "invoice_id", "vendor", "po_id", "invoice_date", "sku",
    "quantity", "unit_price", "po_unit_price", "po_quantity",
    "paid_amount", "expected_amount"
]

def demo_data():
    rows = [
        ["INV-1001","Northstar Supplies","PO-501","2026-10-02","PAPER-A4",100,4.00,4.00,120,400,400],
        ["INV-1002","Northstar Supplies","PO-502","2026-10-03","TONER-01",10,55.00,50.00,10,550,500],
        ["INV-1003","Apex Components","PO-503","2026-10-04","CABLE-22",80,12.00,12.00,80,960,960],
        ["INV-1003","Apex Components","PO-503","2026-10-05","CABLE-22",80,12.00,12.00,80,960,960],
        ["INV-1005","BluePeak Logistics","PO-504","2026-10-06","FREIGHT",1,1800,1500,1,1800,1500],
        ["INV-1006","Northstar Supplies","PO-505","2026-10-07","PAPER-A4",150,4.00,4.00,120,600,480],
        ["INV-1007","Crestline Retail","PO-506","2026-10-08","LABEL-9",50,2.50,2.50,50,125,125],
        ["INV-1008","Apex Components","PO-507","2026-10-09","CABLE-22",20,12.00,12.00,20,240,240],
        ["INV-1009","BluePeak Logistics","PO-508","2026-10-10","FREIGHT",1,1510,1500,1,1510,1500],
        ["INV-1010","Crestline Retail","PO-509","2026-10-11","LABEL-9",45,2.50,2.50,50,112.5,112.5],
        ["INV-1011","Northstar Supplies","PO-510","2026-10-12","TONER-01",8,50,50,8,400,400],
        ["INV-1012","Apex Components","PO-511","2026-10-13","CABLE-22",100,12,12,100,1200,1200],
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

class AuditEngine:
    """Explainable AP controls plus an unsupervised anomaly signal."""

    def run(self, source: pd.DataFrame):
        missing = sorted(set(REQUIRED_COLUMNS) - set(source.columns))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        df = source.copy()
        numeric = ["quantity", "unit_price", "po_unit_price", "po_quantity", "paid_amount", "expected_amount"]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[numeric].isna().any().any():
            raise ValueError("Numeric fields contain blank or non-numeric values.")
        if (df[numeric] < 0).any().any():
            raise ValueError("Numeric fields cannot contain negative values.")

        df["issue_types"] = ""
        df["potential_overpayment"] = 0.0
        df["risk_score"] = 0

        duplicate = df.duplicated(["vendor", "invoice_id"], keep=False)
        df.loc[duplicate, "issue_types"] += "duplicate_invoice;"
        duplicate_copy = df.duplicated(["vendor", "invoice_id"], keep="first")
        duplicate_gap = np.where(duplicate_copy, df["paid_amount"], 0.0)

        price_gap = (df["unit_price"] - df["po_unit_price"]).clip(lower=0) * df["quantity"]
        df.loc[price_gap > 0.01, "issue_types"] += "unit_price_above_po;"

        excess_qty = (df["quantity"] - df["po_quantity"]).clip(lower=0)
        qty_gap = excess_qty * df["po_unit_price"]
        df.loc[excess_qty > 0, "issue_types"] += "quantity_above_po;"

        amount_gap = (df["paid_amount"] - df["expected_amount"]).clip(lower=0)
        df.loc[amount_gap > 0.01, "issue_types"] += "paid_above_expected;"

        # Use the largest supported estimate to avoid double-counting overlapping controls.
        df["potential_overpayment"] = np.maximum.reduce(
            [price_gap.to_numpy(), qty_gap.to_numpy(), amount_gap.to_numpy(), duplicate_gap]
        )

        # Feature engineering for a secondary, unsupervised review signal.
        features = pd.DataFrame({
            "unit_price_variance": (df["unit_price"] - df["po_unit_price"]) / df["po_unit_price"].replace(0, 1),
            "quantity_variance": (df["quantity"] - df["po_quantity"]) / df["po_quantity"].replace(0, 1),
            "payment_variance": (df["paid_amount"] - df["expected_amount"]) / df["expected_amount"].replace(0, 1),
            "invoice_value_log": np.log1p(df["paid_amount"]),
        }).replace([np.inf, -np.inf], 0).fillna(0)

        if len(df) >= 5 and features.nunique().max() > 1:
            scaled = StandardScaler().fit_transform(features)
            model = IsolationForest(contamination="auto", random_state=42)
            model.fit(scaled)
            df["anomaly_signal"] = (model.predict(scaled) == -1)
            df["anomaly_score"] = -model.score_samples(scaled)
        else:
            df["anomaly_signal"] = False
            df["anomaly_score"] = 0.0

        rule_hit = df["issue_types"].ne("")
        df.loc[df["anomaly_signal"], "issue_types"] += "ml_anomaly_signal;"
        df["issue_types"] = df["issue_types"].str.rstrip(";")
        df["risk_score"] = np.where(rule_hit, 55, 0) + np.where(df["anomaly_signal"], 25, 0)
        df["risk_score"] = df["risk_score"].clip(0, 100).astype(int)
        df["risk_level"] = pd.cut(
            df["risk_score"], bins=[-1, 0, 54, 79, 100],
            labels=["No flag", "Low", "Medium", "High"]
        ).astype(str)
        df["issue_types"] = df["issue_types"].str.rstrip(";")
        findings = df[(rule_hit) | (df["anomaly_signal"])].copy()
        findings = findings.sort_values(["risk_score", "potential_overpayment"], ascending=False)

        metrics = {
            "invoice_count": int(len(df)),
            "exposure": round(float(findings["potential_overpayment"].sum()), 2),
            "flagged_count": int(len(findings)),
            "high_risk_vendors": int(findings.loc[findings["risk_level"] == "High", "vendor"].nunique()),
        }
        if "known_issue" in df.columns:
            truth = df["known_issue"].astype(int)
            predicted = (rule_hit | df["anomaly_signal"]).astype(int)
            metrics["evaluation"] = {
                "precision": round(float(precision_score(truth, predicted, zero_division=0)), 3),
                "recall": round(float(recall_score(truth, predicted, zero_division=0)), 3),
                "f1": round(float(f1_score(truth, predicted, zero_division=0)), 3),
            }
        return findings, metrics, df
