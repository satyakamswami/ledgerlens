import json
import os
import requests
import pandas as pd

class FinancialAnalyst:
    """Groq-compatible LLM client with explicit, bounded audit tools."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    @staticmethod
    def _tools():
        return [
            {"type": "function", "function": {
                "name": "vendor_exposure",
                "description": "Calculate flagged invoice count and potential exposure by vendor.",
                "parameters": {"type": "object", "properties": {
                    "vendor": {"type": "string", "description": "Exact vendor name"}
                }, "required": ["vendor"]}
            }},
            {"type": "function", "function": {
                "name": "invoice_evidence",
                "description": "Retrieve the audit evidence for an invoice ID.",
                "parameters": {"type": "object", "properties": {
                    "invoice_id": {"type": "string"}
                }, "required": ["invoice_id"]}
            }},
            {"type": "function", "function": {
                "name": "risk_summary",
                "description": "Summarize the current audit findings and total potential exposure.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }},
        ]

    def _dispatch(self, name, args, records, metrics):
        frame = records
        if name == "vendor_exposure":
            if frame.empty:
                return {"vendor": args["vendor"], "flagged_invoices": 0, "potential_exposure": 0.0}
            subset = frame[frame["vendor"].astype(str).str.casefold() == args["vendor"].casefold()]
            return {"vendor": args["vendor"], "flagged_invoices": len(subset),
                    "potential_exposure": round(float(subset["potential_overpayment"].sum()), 2)}
        if name == "invoice_evidence":
            if frame.empty:
                return []
            subset = frame[frame["invoice_id"].astype(str) == args["invoice_id"]]
            return subset.to_dict(orient="records")
        if name == "risk_summary":
            return {"metrics": metrics, "top_findings": frame.head(10).to_dict(orient="records")}
        return {"error": "Tool not available"}

    def answer(self, question, context, style="zero-shot"):
        if not self.api_key:
            raise RuntimeError("Add GROQ_API_KEY in Render environment variables to enable the AI analyst.")
        examples = {
            "zero-shot": "",
            "one-shot": '\nExample: If asked why an invoice is flagged, cite the exact rule and numeric evidence; never infer intent.',
            "few-shot": (
                '\nExamples:\nQ: Is an anomaly proof of fraud?\nA: No. It is a review signal requiring human validation.'
                '\nQ: How much can be recovered?\nA: Report the calculated potential exposure, not a guaranteed recovery.'
            ),
        }
        system = (
            "You are an accounts-payable audit analyst. Use only supplied records and tool results. "
            "Separate confirmed rule exceptions from unsupervised anomaly signals. Never call an anomaly fraud. "
            "Cite invoice IDs, vendors, rule names and amounts. If evidence is absent, say so. "
            "Do not invent ERP facts, contracts, or recovery outcomes."
            + examples.get(style, "")
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Audit context: " + json.dumps(context, default=str) + "\nQuestion: " + question},
        ]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for _ in range(4):
            payload = {"model": self.model, "messages": messages, "tools": self._tools(), "tool_choice": "auto", "temperature": 0.1}
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                return message.get("content", "The model returned no analysis.")
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                result = self._dispatch(name, args, pd.DataFrame(context["findings"]), context["metrics"])
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, default=str)})
        raise RuntimeError("The analyst exceeded the tool-call limit. Try a narrower question.")
