#!/bin/bash
#SBATCH --job-name=dv_token_stats
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_token_stats_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_token_stats_%j.err
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# ---------------- FILL THESE IN ----------------
#SBATCH --partition=FILL_IN_PARTITION
#SBATCH --time=FILL_IN_TIME
#SBATCH --mem=FILL_IN_MEM
# -----------------------------------------------
# NO GPU NEEDED. Tokenizer only, no model weights. Can also be run on the login
# node if your cluster allows it - it takes seconds.

# Measures the real token-length distribution so max_seq_length is set from data
# rather than carried over. Exits non-zero if any game exceeds --limit; nothing is
# ever dropped or truncated. The previous build silently lost games to over-length
# filtering, so this is a gate, not a filter.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

mkdir -p results/finetuning

for SPEC in \
  "E2B:unsloth/gemma-4-E2B-it-unsloth-bnb-4bit" \
  "E4B:unsloth/gemma-4-E4B-it-unsloth-bnb-4bit" \
  "31B:unsloth/gemma-4-31B-it-unsloth-bnb-4bit" ; do
  TAG="${SPEC%%:*}"
  MODEL="${SPEC#*:}"
  echo
  echo "############ $TAG ############"
  python -m src.finetuning.derivation.token_stats \
    --model_name "$MODEL" \
    --limit 4096 \
    --out "results/finetuning/token_stats_${TAG}.json" || \
    echo "*** $TAG EXCEEDS THE LIMIT - raise max_seq_length in the dv_train scripts ***"
done

echo
echo "=== done. If every model reported 0 games over 4096, leave MAX_SEQ=4096. ==="
echo "=== If not, raise MAX_SEQ in slurm_files/dv_train_*.slurm to clear the max. ==="
