#!/bin/bash
#SBATCH --job-name=dv_eval_adapter
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_eval_adapter_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_eval_adapter_%j.err
#SBATCH --account=thesis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=testing
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=FILL_IN_MEM

# Evaluate ONE trained adapter on the 30 validation games, condition A.
#
#   sbatch --nodelist=<NODE> slurm_files/dv_eval_adapter.slurm <TAG> <ADAPTER_PATH>
#
# e.g.
#   sbatch --nodelist=vgpu9-0 slurm_files/dv_eval_adapter.slurm \
#     31B models/finetuned/gemma-4-31B-derivation-v1/final_adapter
#
# TAG is E2B, E4B or 31B and selects the base checkpoint the adapter attaches to.
# ADAPTER_PATH is any directory the trainer wrote - final or a per-epoch checkpoint.
#
# Comparability with the base diagnostic is the whole point, so all three of the
# things that could move are pinned to the diagnostic's values:
#
#   prompt    val.jsonl's `prompt` field IS condition A - the night actions are
#             included, because that is what build_dataset assembles. Same bytes
#             the diagnostic fed the base model.
#   decoding  temperature 1.0, top_p 0.95, top_k 64, thinking enabled - hardcoded
#             in evaluate_adapter.generate, same as base_diagnostic.
#   budget    max_new_tokens 14000 and max_seq_length 24576, matching the
#             diagnostic rather than evaluate_adapter's own smaller defaults.
#
# Base per-player accuracies to compare against, condition A:
#   E2B 75.3%   E4B 84.7%   31B 100%
#
# 31B 4-bit plus a 24576 KV cache will not fit the 23.5 GB A5000 on vgpu8-0.
# Put 31B on the A6000 or vgpu10.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

if [ "$#" -ne 2 ]; then
  echo "usage: sbatch $0 <TAG: E2B|E4B|31B> <ADAPTER_PATH>" >&2
  exit 2
fi

TAG="$1"
ADAPTER="$2"

case "$TAG" in
  E2B) MODEL="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit" ;;
  E4B) MODEL="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit" ;;
  31B) MODEL="unsloth/gemma-4-31B-it-unsloth-bnb-4bit" ;;
  *) echo "unknown TAG '$TAG' - expected E2B, E4B or 31B" >&2; exit 2 ;;
esac

if [ ! -d "$ADAPTER" ]; then
  echo "adapter path does not exist: $ADAPTER" >&2
  exit 2
fi

VAL="data/processed/jin2024_onuw/sft_derivation_v1/val.jsonl"
STAMP=$(basename "$ADAPTER")
OUT="results/finetuning/adapter_eval_${TAG}_${STAMP}.json"

mkdir -p results/finetuning logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant $TAG, adapter $ADAPTER ==="
nvidia-smi

python -m src.finetuning.evaluate_adapter \
  --model_name "$MODEL" \
  --adapter_path "$ADAPTER" \
  --val_path "$VAL" \
  --max_seq_length 24576 \
  --max_new_tokens 14000 \
  --limit -1 \
  --output_path "$OUT"

echo
echo "=== done. $OUT ==="
echo "=== compare the per-player line against the base: E2B 75.3% / E4B 84.7% / 31B 100% ==="
