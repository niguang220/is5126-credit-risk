#!/usr/bin/env python3
"""
Qwen Scoring — Part 2 Pipeline C
Scores 3,000 loan descriptions via iFlow API (qwen3-max) + consistency check on 100 samples.

Outputs:
  - qwen_cache_3000.json      : {str(idx): {score, reasoning, risk_level, ...}}
  - consistency_cache.json    : {str(idx): [score_r0, ..., score_r4]} (100 samples x 5 runs)

Time estimate: ~3h total (2.5h scoring + 25min consistency)

Usage:
    # First run
    python scripts/run_qwen_3000.py --data-dir ./data --output-dir ./output

    # Resume after crash / Ctrl+C
    python scripts/run_qwen_3000.py --data-dir ./data --output-dir ./output

    # Skip consistency check (scoring only)
    python scripts/run_qwen_3000.py --data-dir ./data --output-dir ./output --no-consistency

Setup on NUS SOC cluster:
    1. ssh xlogin.comp.nus.edu.sg
    2. conda activate is5126
    3. export IFLOW_API_KEY="your_key_here"
    4. tmux new -s qwen3k
    5. python scripts/run_qwen_3000.py --data-dir ~/is5126/data --output-dir ~/is5126/output
    6. Ctrl+B, D to detach

    OR: submit via SLURM:
    sbatch scripts/submit_3000.sh
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────
API_BASE_URL    = "https://apis.iflow.cn/v1"
MODEL           = "qwen3-max"
TEMPERATURE     = 0.0
MAX_TOKENS      = 512
REQUEST_TIMEOUT = 60       # seconds
SEED            = 42
CONSISTENCY_N   = 100      # how many samples for consistency check
CONSISTENCY_RUNS = 5       # runs per sample

SYSTEM_PROMPT = """You are a credit risk analyst at a P2P lending company.
Assess default risk SOLELY from the loan description provided.

Respond in EXACT JSON format and nothing else:
{
  "default_probability": <float between 0.0 and 1.0>,
  "risk_level": "<LOW or MEDIUM or HIGH>",
  "key_risk_factors": ["<factor1>", "<factor2>"],
  "key_protective_factors": ["<factor1>", "<factor2>"],
  "reasoning": "<2-3 sentence analysis>"
}

Rules:
- ONLY reference content explicitly present in the loan description
- Do NOT invent or assume information not in the description
- Base rate of default in this portfolio is approximately 15%
- LOW risk: 0.01-0.10, MEDIUM risk: 0.10-0.30, HIGH risk: 0.30+
- Respond with ONLY the JSON object, no additional text"""


# ── Helpers ───────────────────────────────────────────────────────────

def setup_logging(output_dir: Path):
    log_file = output_dir / f"qwen3k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging to {log_file}")


def strip_think_tags(content: str) -> str:
    """Strip <think>...</think> chain-of-thought emitted by Qwen3-Max."""
    match = re.search(r"</think>\s*(.*)", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def parse_qwen_response(content: str) -> dict:
    """Parse JSON from Qwen response. Falls back gracefully on error."""
    content = strip_think_tags(content)

    # Strip markdown code fences if present
    md_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if md_match:
        content = md_match.group(1).strip()

    try:
        parsed = json.loads(content)
        score = float(parsed.get("default_probability", -1))
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "risk_level": parsed.get("risk_level", ""),
            "key_risk_factors": parsed.get("key_risk_factors", []),
            "key_protective_factors": parsed.get("key_protective_factors", []),
            "reasoning": parsed.get("reasoning", ""),
            "parse_error": False,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        # Last-resort: try to extract a float from the raw content
        float_match = re.search(r"\b(0\.\d+|1\.0)\b", content)
        score = float(float_match.group(1)) if float_match else float("nan")
        return {
            "score": score,
            "raw_response": content[:500],
            "parse_error": True,
        }


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(cached: dict, path: Path):
    """Atomic write: tmp file then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cached, f)
    tmp.replace(path)


# ── LLM caller ────────────────────────────────────────────────────────

def call_qwen(client, user_content: str, max_retries: int = 5) -> dict:
    """Streaming call with exponential backoff on retryable errors."""
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
            return parse_qwen_response(content)

        except json.JSONDecodeError:
            return {"score": float("nan"), "raw_response": content[:500], "parse_error": True}

        except Exception as e:
            err = str(e).lower()
            retryable = any(k in err for k in ["rate", "limit", "429", "500", "502", "503", "timeout", "connection"])
            if retryable and attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 80)  # 5, 10, 20, 40, 80s
                logging.warning(f"Retryable error (attempt {attempt+1}/{max_retries}), waiting {wait}s: {e}")
                time.sleep(wait)
            else:
                logging.error(f"Failed after {attempt+1} attempts: {e}")
                return {"score": float("nan"), "parse_error": True, "error_type": str(e)}

    return {"score": float("nan"), "parse_error": True, "error_type": "max_retries_exhausted"}


# ── Main scoring loop ─────────────────────────────────────────────────

def run_scoring(df: pd.DataFrame, client, output_dir: Path, delay: float, checkpoint_every: int):
    """Score all 3,000 desc texts. Resumes from checkpoint automatically."""
    checkpoint_path = output_dir / "qwen_cache_3000.json"
    cached = load_checkpoint(checkpoint_path)

    all_idxs = [str(i) for i in df.index.tolist()]
    todo_idxs = [i for i in all_idxs if i not in cached]
    logging.info(f"Scoring: total={len(all_idxs)} cached={len(cached)} remaining={len(todo_idxs)}")

    if not todo_idxs:
        logging.info("All 3,000 already scored!")
        return cached

    todo_df = df.loc[[int(i) for i in todo_idxs]]
    errors = 0
    start = time.time()

    for i, (idx, row) in enumerate(tqdm(todo_df.iterrows(), total=len(todo_df), desc="Qwen scoring")):
        desc = str(row.get("desc", ""))[:500]
        user_msg = f"Loan description: '{desc}'\n\nAssess default risk:"
        result = call_qwen(client, user_msg)
        cached[str(idx)] = result
        if result.get("parse_error"):
            errors += 1

        time.sleep(delay)

        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(cached, checkpoint_path)
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = timedelta(seconds=int((len(todo_df) - i - 1) / rate)) if rate > 0 else "?"
            logging.info(
                f"Checkpoint {len(cached)}/{len(all_idxs)} "
                f"({len(cached)/len(all_idxs)*100:.1f}%) | errors={errors} | ETA={eta}"
            )

    save_checkpoint(cached, checkpoint_path)
    elapsed = time.time() - start
    logging.info(f"Scoring done: {len(todo_df)} in {timedelta(seconds=int(elapsed))} | errors={errors}")
    return cached


# ── Consistency check ─────────────────────────────────────────────────

def run_consistency(df: pd.DataFrame, client, output_dir: Path, delay: float):
    """
    Run 5 independent scoring passes on 100 random samples.
    Saves consistency_cache.json: {str(idx): [score_r0, ..., score_r4]}
    """
    checkpoint_path = output_dir / "consistency_cache.json"
    cached = load_checkpoint(checkpoint_path)

    # Pick 100 samples (reproducible)
    rng = np.random.default_rng(SEED)
    sample_idxs = rng.choice(df.index.tolist(), size=min(CONSISTENCY_N, len(df)), replace=False)
    sample_idxs = [str(i) for i in sample_idxs]

    # Resume: find which (idx, run) pairs are already done
    todo = []
    for idx in sample_idxs:
        existing = cached.get(idx, [])
        for run in range(CONSISTENCY_RUNS):
            if run >= len(existing):
                todo.append((idx, run))

    logging.info(f"Consistency: {len(sample_idxs)} samples x {CONSISTENCY_RUNS} runs | remaining={len(todo)}")

    if not todo:
        logging.info("Consistency check already complete!")
        return cached

    for idx, run in tqdm(todo, desc="Consistency"):
        desc = str(df.loc[int(idx)].get("desc", ""))[:500]
        user_msg = f"Loan description: '{desc}'\n\nAssess default risk:"
        result = call_qwen(client, user_msg)
        score = result.get("score", float("nan"))

        if idx not in cached:
            cached[idx] = []
        # Pad list to correct length if resuming mid-sample
        while len(cached[idx]) < run:
            cached[idx].append(float("nan"))
        if run < len(cached[idx]):
            cached[idx][run] = score
        else:
            cached[idx].append(score)

        time.sleep(delay)

    save_checkpoint(cached, checkpoint_path)
    logging.info(f"Consistency check complete. Saved to {checkpoint_path}")
    return cached


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen scoring for IS5126 Part 2 Pipeline C")
    parser.add_argument("--data-dir", required=True, help="Dir containing sample_3000.parquet")
    parser.add_argument("--output-dir", required=True, help="Dir for cache JSON files and logs")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint every N calls")
    parser.add_argument("--no-consistency", action="store_true", help="Skip consistency check")
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

    # Quick connectivity test
    logging.info("Testing API connection...")
    test = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Say OK"}], max_tokens=5
    )
    logging.info(f"API OK: {test.choices[0].message.content.strip()}")

    # Load data
    parquet_path = data_dir / "sample_3000.parquet"
    logging.info(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded {len(df)} rows | default rate: {df['default'].mean():.4f}")

    logging.info("=" * 60)
    logging.info("PHASE 1: Main scoring (3,000 desc texts)")
    logging.info("=" * 60)
    cached = run_scoring(df, client, output_dir, delay=args.delay, checkpoint_every=args.checkpoint_every)

    valid = sum(1 for v in cached.values() if not v.get("parse_error", True))
    logging.info(f"Scoring complete: {valid}/{len(cached)} valid ({valid/len(cached)*100:.1f}%)")

    if not args.no_consistency:
        logging.info("=" * 60)
        logging.info(f"PHASE 2: Consistency check ({CONSISTENCY_N} samples x {CONSISTENCY_RUNS} runs)")
        logging.info("=" * 60)
        run_consistency(df, client, output_dir, delay=args.delay)

    logging.info("\nDone! Upload output JSON files to Google Drive for NB05.")
    logging.info(f"  {output_dir}/qwen_cache_3000.json")
    if not args.no_consistency:
        logging.info(f"  {output_dir}/consistency_cache.json")


if __name__ == "__main__":
    main()
