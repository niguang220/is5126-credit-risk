#!/usr/bin/env python3
"""
Qwen Correction — Part 3
LLM second-pass review on borderline ML cases (full structured profile, no desc).

Outputs:
  - qwen_cache_200.json : {str(idx): {decision, reasoning, raw, parse_error}}

Time estimate: ~10 minutes (~100 borderline cases)

Usage:
    python scripts/run_qwen_correction.py --data-dir ./data --output-dir ./output

Prerequisites (run in Colab NB06 first):
    - sample_200.parquet   : 200 samples with ML predictions and features
    - shap_context_200.json: top SHAP factors per case
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────
API_BASE_URL    = "https://apis.iflow.cn/v1"
MODEL           = "qwen3-max"
TEMPERATURE     = 0.0
MAX_TOKENS      = 256
REQUEST_TIMEOUT = 60
BORDERLINE_LOW  = 0.3
BORDERLINE_HIGH = 0.7

# LC feature labels for human-readable serialization
FEATURE_LABELS = {
    "loan_amnt":              "Loan Amount ($)",
    "int_rate":               "Interest Rate (%)",
    "installment":            "Monthly Installment ($)",
    "grade_num":              "Grade (1=A to 7=G)",
    "sub_grade_num":          "Sub-grade (numeric)",
    "emp_length":             "Employment Length (years)",
    "home_ownership":         "Home Ownership",
    "annual_inc":             "Annual Income ($)",
    "verification_status":    "Income Verification Status",
    "purpose":                "Loan Purpose",
    "addr_state":             "State",
    "dti":                    "Debt-to-Income Ratio",
    "delinq_2yrs":            "Delinquencies (past 2 years)",
    "inq_last_6mths":         "Credit Inquiries (past 6 months)",
    "open_acc":               "Open Credit Accounts",
    "pub_rec":                "Public Records (derogatory)",
    "revol_bal":              "Revolving Balance ($)",
    "revol_util":             "Revolving Credit Utilization (%)",
    "total_acc":              "Total Credit Accounts",
    "loan_to_income":         "Loan-to-Income Ratio",
    "installment_to_income":  "Installment-to-Income Ratio",
    "delinq_per_year":        "Delinquency Rate (per year)",
    "inq_per_open_acc":       "Inquiries per Open Account",
    "is_long_term":           "Long-term Loan (60 months)",
    "term":                   "Loan Term",
}

SYSTEM_PROMPT = """You are a senior credit analyst reviewing a borderline loan application.
The ML model is uncertain about this case (probability near 0.5).
Your job is to apply expert credit judgment based on the full applicant profile.

Respond in EXACT format:
Line 1: DEFAULT or NO DEFAULT
Line 2: One sentence explaining the most decisive factor.

No other text. No JSON. No preamble."""


# ── Helpers ───────────────────────────────────────────────────────────

def setup_logging(output_dir: Path):
    log_file = output_dir / f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def strip_think_tags(content: str) -> str:
    match = re.search(r"</think>\s*(.*)", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def parse_correction_response(content: str) -> dict:
    content = strip_think_tags(content)
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    decision_line = lines[0].upper() if lines else ""
    reasoning = lines[1] if len(lines) > 1 else ""

    if "NO DEFAULT" in decision_line:
        decision = "NO DEFAULT"
    elif "DEFAULT" in decision_line:
        decision = "DEFAULT"
    else:
        decision = None

    return {
        "decision": decision,
        "reasoning": reasoning,
        "raw": content[:500],
        "parse_error": decision is None,
    }


def serialize_profile(row: pd.Series, shap_context: dict | None = None,
                      ml_proba: float | None = None) -> str:
    lines = ["=== Loan Application Profile ==="]
    for feat, label in FEATURE_LABELS.items():
        val = row.get(feat)
        if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "float" and val != val):
            continue
        if isinstance(val, float):
            val_str = f"{val:,.2f}"
        else:
            val_str = str(val)
        lines.append(f"- {label}: {val_str}")

    if ml_proba is not None:
        lines.append(f"\nML Assessment: {ml_proba:.1%} default probability (UNCERTAIN — borderline case)")

    if shap_context:
        risk_factors = shap_context.get("risk_factors", [])
        protective_factors = shap_context.get("protective_factors", [])
        if risk_factors:
            lines.append(f"Top ML risk signals: {', '.join(risk_factors)}")
        if protective_factors:
            lines.append(f"Top ML protective signals: {', '.join(protective_factors)}")

    return "\n".join(lines)


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(cached: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cached, f)
    tmp.replace(path)


# ── LLM caller ────────────────────────────────────────────────────────

def call_qwen(client, user_content: str, max_retries: int = 5) -> dict:
    content = ""
    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            content = "".join(chunks)
            return parse_correction_response(content)

        except Exception as e:
            err = str(e).lower()
            retryable = any(k in err for k in ["rate", "limit", "429", "500", "502", "503", "timeout", "connection"])
            if retryable and attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 80)
                logging.warning(f"Retryable error (attempt {attempt+1}), waiting {wait}s: {e}")
                time.sleep(wait)
            else:
                logging.error(f"Failed after {attempt+1} attempts: {e}")
                return {"decision": None, "reasoning": "", "raw": "", "parse_error": True, "error_type": str(e)}

    return {"decision": None, "parse_error": True, "error_type": "max_retries_exhausted"}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen correction for IS5126 Part 3")
    parser.add_argument("--data-dir", required=True, help="Dir with sample_200.parquet and shap_context_200.json")
    parser.add_argument("--output-dir", required=True, help="Dir for output cache and logs")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    api_key = os.environ.get("IFLOW_API_KEY", "")
    if not api_key:
        raise ValueError("IFLOW_API_KEY not set. Run: export IFLOW_API_KEY='your_key'")

    import openai
    client = openai.OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # Load data
    df = pd.read_parquet(data_dir / "sample_200.parquet")
    logging.info(f"Loaded sample_200: {len(df)} rows")

    shap_path = data_dir / "shap_context_200.json"
    if shap_path.exists():
        with open(shap_path) as f:
            shap_context = json.load(f)
        logging.info(f"Loaded shap_context_200: {len(shap_context)} entries")
    else:
        shap_context = {}
        logging.warning("shap_context_200.json not found — SHAP context will be omitted from prompts")

    # Filter to borderline only
    ml_col = "ml_proba"
    if ml_col not in df.columns:
        raise ValueError(f"Column '{ml_col}' not found. Run NB06 Colab cells first to compute ML predictions.")

    borderline_mask = df[ml_col].between(BORDERLINE_LOW, BORDERLINE_HIGH)
    borderline_df = df[borderline_mask].copy()
    logging.info(f"Borderline cases: {len(borderline_df)} / {len(df)} "
                 f"(ML prob in [{BORDERLINE_LOW}, {BORDERLINE_HIGH}])")

    # Load checkpoint
    checkpoint_path = output_dir / "qwen_cache_200.json"
    cached = load_checkpoint(checkpoint_path)
    todo_df = borderline_df[~borderline_df.index.astype(str).isin(cached.keys())]
    logging.info(f"Remaining to score: {len(todo_df)}")

    if todo_df.empty:
        logging.info("All borderline cases already scored!")
    else:
        errors = 0
        for idx, row in tqdm(todo_df.iterrows(), total=len(todo_df), desc="Qwen correction"):
            ml_proba = float(row[ml_col])
            ctx = shap_context.get(str(idx))
            prompt = serialize_profile(row, shap_context=ctx, ml_proba=ml_proba)
            result = call_qwen(client, prompt)
            result["ml_proba"] = ml_proba
            result["y_true"] = int(row.get("default", -1))
            cached[str(idx)] = result
            if result.get("parse_error"):
                errors += 1
            time.sleep(args.delay)

        save_checkpoint(cached, checkpoint_path)
        logging.info(f"Done: {len(todo_df)} scored | errors={errors}")

    # Summary
    decisions = [v.get("decision") for v in cached.values() if not v.get("parse_error")]
    n_default = sum(1 for d in decisions if d == "DEFAULT")
    n_no_default = sum(1 for d in decisions if d == "NO DEFAULT")
    logging.info(f"\nSummary: DEFAULT={n_default} | NO DEFAULT={n_no_default} | "
                 f"parse errors={sum(1 for v in cached.values() if v.get('parse_error'))}")
    logging.info(f"\nSaved to {checkpoint_path}")
    logging.info("Upload to Google Drive then load in NB06.")


if __name__ == "__main__":
    main()
