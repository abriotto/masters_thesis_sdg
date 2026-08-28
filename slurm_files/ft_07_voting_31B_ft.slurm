#!/bin/bash
#SBATCH --job-name=vote_31B_ft
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/vote_31B_ft_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/vote_31B_ft_%j.err
#SBATCH --partition=owner1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# Seed index. Submit as:  sbatch --export=ALL,RUN=1 slurm_files/ft_07_voting_31B_ft.slurm
RUN="${RUN:-1}"

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) - 31B finetuned, run ${RUN} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Generation budget RAISED over the 31B base v4 runs (16384 / 8000-12000) because the
# finetuned model thinks longer (16,155 vs 9,893 chars in the probe). Caps were
# non-binding in the base arm - 0 of 573 games reached 95% of their cap, max output
# 9061 - so a higher cap can only prevent truncation, never advantage this arm.
# Verify 0 truncations post hoc and report the caps for both arms.
# Per-game JSON is written as each game finishes and existing files are skipped
# without --overwrite, so a walltime kill costs only the game in flight: resubmit
# the identical command to resume.
python -m src.voting.run_llm_votes \
  --model_name unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --adapter_path models/finetuned/gemma-4-31B-role-inference-traced/final_adapter \
  --prompt_path src/prompts/voting_prompt_v4.txt \
  --prompt_version "v4_run_${RUN}" \
  --gemma_enable_thinking \
  --save_internal_thoughts \
  --max_seq_length 24576 \
  --max_new_tokens 14000

echo "=== done: 31B finetuned run ${RUN} ==="
