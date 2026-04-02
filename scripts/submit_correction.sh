#!/bin/bash
#SBATCH --job-name=qwen_correction
#SBATCH --partition=short
#SBATCH --time=0-01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --exclude=xcng0,xcng1
#SBATCH --output=logs/correction_%j.out
#SBATCH --error=logs/correction_%j.err

# ── Usage ────────────────────────────────────────────────────────────
# Prerequisites (run in Colab NB06 first to generate these files):
#   - sample_200.parquet
#   - shap_context_200.json
#
# 1. Upload both files to ~/is5126/data/
# 2. Get fresh API key at https://platform.iflow.cn
# 3. Edit IFLOW_API_KEY below
# 4. sbatch scripts/submit_correction.sh
#
# Or just run locally (only ~10 min):
#   export IFLOW_API_KEY="your_key"
#   python scripts/run_qwen_correction.py --data-dir ./data --output-dir ./output
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# !! Replace with your fresh API key !!
export IFLOW_API_KEY="REPLACE_WITH_YOUR_KEY"

REPO_DIR="$HOME/is5126"
DATA_DIR="$REPO_DIR/data"
OUTPUT_DIR="$REPO_DIR/output"

mkdir -p "$OUTPUT_DIR" logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate is5126

echo "Job started: $(date)"
echo "Node: $(hostname)"

cd "$REPO_DIR"
python scripts/run_qwen_correction.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --delay 0.5

echo "Job finished: $(date)"
echo "Output:"
ls -lh "$OUTPUT_DIR"/qwen_cache_200.json 2>/dev/null || echo "qwen_cache_200.json not found"
