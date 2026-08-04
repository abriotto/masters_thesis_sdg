#!/bin/bash
#SBATCH --job-name=ft_role
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_%j.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=thesis

# Step 5: the real run. Only submit this once ft_01 and ft_02 came back clean.
#
# Estimated 2-3h for 3 epochs over 99 examples (~74 optimizer steps), plus ~20 min
# for the post-training smoke test, which generates at inference speed.
# --time is set to 8h as headroom; the `debug` partition may cap lower than that,
# so check your partition limits before submitting.

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# conda activate <your_env>   # uncomment if the cluster needs it

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -m src.finetuning.train_role_inference \
  --output_dir models/finetuned/gemma-4-31B-role-inference-v1 \
  --num_train_epochs 3 \
  --smoke_test_samples 3

echo
echo "=== training finished ==="
echo "Adapter:     models/finetuned/gemma-4-31B-role-inference-v1/final_adapter"
echo "Checkpoints: models/finetuned/gemma-4-31B-role-inference-v1/checkpoint-*"
echo
echo "Read the smoke test above BEFORE launching the voting reruns:"
echo "if the thought channel is missing, use an earlier checkpoint instead."
