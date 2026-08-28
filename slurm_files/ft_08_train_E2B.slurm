#!/bin/bash
#SBATCH --job-name=ft_role_E2B
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_E2B_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_E2B_%j.err
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# conda activate <your_env>   # uncomment if the cluster needs it

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# max_seq_length matches the E4B run so the three models are trained under the
# same budget. The E2B dataset's longest example is 5276 tokens, so nothing is
# truncated at 8192.
MODEL=unsloth/gemma-4-E2B-it-unsloth-bnb-4bit
DATA=data/processed/jin2024_onuw/sft_role_inference_traced_E2B
OUT=models/finetuned/gemma-4-E2B-role-inference-traced

# Loss-mask gate. Prints the exact supervised span: it must be the gold JSON plus
# the turn-end token and must NOT contain the thought-channel markers, or the
# rationale is being trained instead of used as context. --output_dir is passed
# here only so the log names this run rather than the script's stale default.
python -m src.finetuning.train_role_inference \
  --model_name "$MODEL" \
  --train_path "$DATA/train.jsonl" \
  --val_path "$DATA/val.jsonl" \
  --output_dir "$OUT" \
  --response_part '<channel|>' \
  --enable_thinking_in_prompt \
  --max_seq_length 8192 \
  --inspect_only

python -m src.finetuning.train_role_inference \
  --model_name "$MODEL" \
  --train_path "$DATA/train.jsonl" \
  --val_path "$DATA/val.jsonl" \
  --output_dir "$OUT" \
  --response_part '<channel|>' \
  --enable_thinking_in_prompt \
  --max_seq_length 8192 \
  --num_train_epochs 3 \
  --smoke_test_samples 3
