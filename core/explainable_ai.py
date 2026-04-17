"""
explainable_ai.py
=================
Sends pipeline metrics and preprocessing metadata to the Gemini 2.0 Flash API,
and gets back a plain-English business explanation saved as explainable_ai/explain.md.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # DO NOT PUSH REAL KEYS TO GITHUB
OUTPUT_DIR = "explainable_ai"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "explain.md")


def build_prompt(metrics_summary: dict, preprocessing_meta: dict, config: dict) -> str:
    """
    Build a rich, context-aware prompt for Gemini so it can write a
    retailer-friendly explanation of what the forecasting pipeline did.
    """
    target_col = config.get("data", {}).get("target_col", "sales")
    num_files = len(config.get("data", {}).get("files", []))

    rows_loaded = preprocessing_meta.get("rows_loaded", "N/A")
    rows_after_outliers = preprocessing_meta.get("rows_after_outliers", "N/A")
    outliers_removed = preprocessing_meta.get("outliers_removed", 0)
    cols_dropped = preprocessing_meta.get("columns_dropped", [])
    neg_target_dropped = preprocessing_meta.get("negative_target_dropped", 0)
    imputation_strategy = preprocessing_meta.get("numeric_imputation", "median")
    features_selected = preprocessing_meta.get("features_selected", "N/A")
    features_total = preprocessing_meta.get("features_total", "N/A")
    models_skipped = preprocessing_meta.get("models_skipped", [])
    routing_reason = preprocessing_meta.get("model_selection_reasoning", "The system determined the best models based on standard scaling principles.")

    # Build model metrics table text
    model_lines = []
    best_model = None
    best_acc = -1
    for model_name, m in metrics_summary.items():
        acc = m.get("accuracy", 0)
        mae = m.get("mae", "N/A")
        rmse = m.get("rmse", "N/A")
        model_lines.append(
            f"  - **{model_name.upper()}**: Accuracy = {acc:.2f}% | MAE = {mae:.4f} | RMSE = {rmse:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            best_model = model_name

    models_text = "\n".join(model_lines)

    skipped_text = (
        f"  - Models intentionally skipped: {', '.join(models_skipped)}\n  - Intelligent Routing Rationale: {routing_reason}"
        if models_skipped
        else f"  - All requested models were trained.\n  - Intelligent Routing Rationale: {routing_reason}"
    )

    cols_text = (
        f"  - Columns auto-dropped (>30% missing): {', '.join(cols_dropped)}"
        if cols_dropped
        else "  - No columns were dropped for excessive missing values."
    )

    prompt = f"""
You are a Senior Retail Business Intelligence Consultant. You have just run an automated
AI-powered sales forecasting analysis for a retail business owner. Your job is to explain
the results in plain, friendly English — no technical jargon, no math formulas.

Write a detailed report in Markdown format that a retail store owner (with no data science
background) can read and immediately understand. Make it warm, confident, and actionable.
Structure your report with the following sections:

1. **Executive Summary** — What did we do and what did we find? (2–3 sentences, very simple)
2. **Your Data at a Glance** — Describe the dataset we worked with
3. **How We Cleaned Your Data** — Explain what data issues we found and how we fixed them
4. **The Forecasting Models We Ran** — Plain-English description of each model and what it found
5. **Which Model Won & Why** — Tell the retailer which model to trust and why
6. **What This Means For Your Business** — Practical takeaways and actionable advice
7. **Confidence Level** — Give a simple 1–10 confidence score and explain why

Here is the raw data from the pipeline run to base your explanation on:

---
**Pipeline Run:** {datetime.now().strftime('%B %d, %Y at %H:%M')}
**Target being forecasted:** `{target_col}` (the key sales or revenue metric)
**Number of data files merged:** {num_files}

**Data Volume:**
  - Total rows loaded: {rows_loaded:,} records
  - Rows after outlier removal: {rows_after_outliers:,} records
  - Outlier records removed: {outliers_removed:,} (extreme anomalies were excluded)
  - Records with invalid/negative {target_col} removed: {neg_target_dropped:,}

**Data Quality:**
{cols_text}
  - Missing values filled using: {imputation_strategy} (automatic best-guess filling)

**Feature Intelligence:**
  - Total data columns analyzed: {features_total}
  - Key columns selected by the AI as most important for forecasting: {features_selected}
{skipped_text}

**Model Performance Results:**
{models_text}

**Best performing model:** {best_model.upper() if best_model else 'N/A'} with {best_acc:.2f}% accuracy

---

Write the report now. Be friendly, specific, and reassuring. Avoid all technical terms unless
you immediately explain them in brackets. The retailer should feel empowered and informed after
reading this.
"""
    return prompt.strip()


def call_gemini_api(prompt: str) -> str:
    """Make a REST POST call to Gemini 2.0 Flash and return the text response."""
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    req = urllib.request.Request(GEMINI_ENDPOINT, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        logging.error(f"Gemini API HTTP error {e.code}: {error_body}")
        return f"[Gemini API Error {e.code}]: Could not generate explanation. Raw error: {error_body}"
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return f"[Gemini API Error]: {str(e)}"


def generate_explanation(metrics_summary: dict, preprocessing_meta: dict, config: dict):
    """
    Main entry point — called by the Evaluator at the end of the pipeline.
    Builds the prompt, calls Gemini, and saves explain.md.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info("Generating AI-powered retail explanation via Gemini...")

    prompt = build_prompt(metrics_summary, preprocessing_meta, config)
    explanation = call_gemini_api(prompt)

    # Write the output file
    header = f"""# 🤖 AI-Powered Forecasting Report
> *Generated by the Universal Retail Forecasting Framework using Gemini 2.0 Flash*
> *{datetime.now().strftime('%B %d, %Y at %H:%M')}*

---

"""
    full_content = header + explanation

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_content)

    logging.info(f"Explanation saved → {OUTPUT_FILE}")
    return OUTPUT_FILE
