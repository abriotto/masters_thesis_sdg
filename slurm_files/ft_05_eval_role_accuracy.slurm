#!/bin/bash
#SBATCH --job-name=ft_eval_role
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_eval_role_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_eval_role_%j.err
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Role accuracy on the 21 held-out validation episodes, BASE vs finetuned.
#
# Submit one job per model:
#   for M in E2B E4B 31B; do sbatch --export=ALL,MODEL=$M slurm_files/ft_05_eval_role_accuracy.slurm; done
#
# Both arms of a model run in the same job, on the same GPU, with identical
# --val_path / --max_seq_length / --max_new_tokens, so the only difference
# between them is the adapter. The UNTRACED val set is used throughout, so
# every condition sees identical prompts and gold labels.

MODEL="${MODEL:-E2B}"

case "$MODEL" in
  31B) BASE=unsloth/gemma-4-31B-it-unsloth-bnb-4bit ;;
  E4B) BASE=unsloth/gemma-4-E4B-it-unsloth-bnb-4bit ;;
  E2B) BASE=unsloth/gemma-4-E2B-it-unsloth-bnb-4bit ;;
  *)   echo "unknown MODEL=$MODEL (expected 31B, E4B or E2B)"; exit 1 ;;
esac

ADAPTER="models/finetuned/gemma-4-${MODEL}-role-inference-traced/final_adapter"
VAL="data/processed/jin2024_onuw/sft_role_inference/val.jsonl"

if [ ! -d "$ADAPTER" ]; then
  echo "adapter not found: $ADAPTER"; exit 1
fi

mkdir -p results/finetuning

echo "############ $MODEL BASE (no adapter) ############"
python -m src.finetuning.evaluate_adapter \
  --model_name "$BASE" \
  --val_path "$VAL" \
  --max_seq_length 12000 \
  --max_new_tokens 10000 \
  --output_path "results/finetuning/eval_${MODEL}_base_val.json"

echo "############ $MODEL FINETUNED (traced adapter) ############"
python -m src.finetuning.evaluate_adapter \
  --model_name "$BASE" \
  --adapter_path "$ADAPTER" \
  --val_path "$VAL" \
  --max_seq_length 12000 \
  --max_new_tokens 10000 \
  --output_path "results/finetuning/eval_${MODEL}_traced_val.json"

echo "=== done: $MODEL ==="
