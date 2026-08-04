#!/bin/bash
#SBATCH --job-name=ft_smoke
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_smoke_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_smoke_%j.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --account=thesis

# Steps 3-4 in ONE job, to pay the queue wait once.
#   step 3: load the model, attach LoRA, print the exact token-level loss mask, exit.
#   step 4: train 2 steps on 4 examples to exercise the loop end to end.
# `set -e` matters here: if step 3 fails there is no point running step 4.

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# conda activate <your_env>   # uncomment if the cluster needs it

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "=== STEP 3: inspect the real loss mask ==="
python -m src.finetuning.train_role_inference --inspect_only

echo
echo "=== STEP 4: smoke-train the loop (throwaway adapter) ==="
python -m src.finetuning.train_role_inference \
  --limit_train_examples 4 \
  --max_steps 2 \
  --smoke_test_samples 1 \
  --output_dir models/finetuned/_smoke_test

echo
echo "=== smoke test finished ==="
echo "The adapter in models/finetuned/_smoke_test is trained on 4 examples for 2 steps."
echo "It is meaningless - delete it so it cannot end up in an evaluation:"
echo "  rm -rf models/finetuned/_smoke_test"
