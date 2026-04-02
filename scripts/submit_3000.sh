#!/bin/bash
#SBATCH --job-name=qwen3k
#SBATCH --partition=long
#SBATCH --time=0-08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --exclude=xcng0,xcng1
#SBATCH --output=logs/qwen3k_%j.out
#SBATCH --error=logs/qwen3k_%j.err

# ── Usage ────────────────────────────────────────────────────────────
# 1. Upload sample_3000.parquet to ~/is5126/data/
# 2. Get fresh API key at https://platform.iflow.cn (expires every 7 days)
# 3. Edit IFLOW_API_KEY below
# 4. sbatch scripts/submit_3000.sh
# 5. squeue -u $USER  (check job status)
# 6. tail -f logs/qwen3k_<jobid>.out  (follow progress)
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# !! Replace with your fresh API key !!
export IFLOW_API_KEY="REPLACE_WITH_YOUR_KEY"

# Paths (adjust if needed)
REPO_DIR="$HOME/is5126"
DATA_DIR="$REPO_DIR/data"
OUTPUT_DIR="$REPO_DIR/output"

mkdir -p "$OUTPUT_DIR" logs

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate is5126

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"

cd "$REPO_DIR"
python scripts/run_qwen_3000.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --delay 0.5 \
    --checkpoint-every 100

echo "Job finished: $(date)"
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON files found"
