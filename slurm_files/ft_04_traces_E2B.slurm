#!/bin/bash
#SBATCH --job-name=ft_traces_E2B
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_traces_E2B_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_traces_E2B_%j.err
#SBATCH --partition=owner1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# conda activate <your_env>   # uncomment if the cluster needs it

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Base-model traces. Resumable: re-running skips episodes already in the output
# file, so a walltime kill costs only the episode in flight.
python -m src.finetuning.generate_traces --model_name unsloth/gemma-4-E2B-it-unsloth-bnb-4bit --output_path data/processed/jin2024_onuw/traces/role_inference_traces_E2B.jsonl --max_seq_length 8192 --max_new_tokens 6000

# Only reached if the traces above completed (set -e). Markers are the family
# defaults, confirmed for E2B by the ~185/191 baseline thoughts parsed with the
# same <|channel>thought / <channel|> regex in model_utils._parse_gemma_thinking.
python -m src.finetuning.build_traced_dataset --traces_path data/processed/jin2024_onuw/traces/role_inference_traces_E2B.jsonl --output_dir data/processed/jin2024_onuw/sft_role_inference_traced_E2B --model_name unsloth/gemma-4-E2B-it-unsloth-bnb-4bit --max_total_tokens 7800
