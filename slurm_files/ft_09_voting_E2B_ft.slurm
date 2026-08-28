#!/bin/bash
#SBATCH --job-name=vote_E2B_ft
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/vote_E2B_ft_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/vote_E2B_ft_%j.err
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# Seed index. Submit as:  sbatch --export=ALL,RUN=1 slurm_files/ft_09_voting_E2B_ft.slurm
RUN="${RUN:-1}"

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) - E2B finetuned, run ${RUN} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Generation config is an EXACT match to the E2B base v4 runs (16384 / 10000).
# Verified non-binding: base max output over 3x191 games was 1541 tokens.
# E2B fails to open a thought channel in ~5 of 191 games per run at baseline;
# that is a property of the model, not of this run, so do not treat it as a fault.
# Per-game JSON is written as each game finishes and existing files are skipped
# without --overwrite, so a walltime kill costs only the game in flight: resubmit
# the identical command to resume.
python -m src.voting.run_llm_votes \
  --model_name unsloth/gemma-4-E2B-it-unsloth-bnb-4bit \
  --adapter_path models/finetuned/gemma-4-E2B-role-inference-traced/final_adapter \
  --prompt_path src/prompts/voting_prompt_v4.txt \
  --prompt_version "v4_run_${RUN}" \
  --gemma_enable_thinking \
  --save_internal_thoughts \
  --max_seq_length 16384 \
  --max_new_tokens 10000

echo "=== done: E2B finetuned run ${RUN} ==="
