#!/bin/bash
#SBATCH --job-name=ft_dryrun
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_dryrun_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_dryrun_%j.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --account=thesis

# Steps 1-2: environment check + tokenizer-only dry run. No GPU needed.
# If your partition refuses jobs without a GPU, add: #SBATCH --gres=gpu:1

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# conda activate <your_env>   # uncomment if the cluster needs it

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="

echo
echo "=== STEP 1: dependency check ==="
python - <<'PY'
import importlib
for name in ["torch", "transformers", "trl", "peft", "datasets", "unsloth"]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:14s} {getattr(mod, '__version__', 'unknown')}")
    except ImportError as exc:
        print(f"  {name:14s} MISSING  ({exc})")
PY

echo
echo "=== STEP 2: dry run (tokenizer only) ==="
python -m src.finetuning.train_role_inference --dry_run

echo
echo "=== dry run finished ==="
echo "Check above: chat template resolved, response marker found,"
echo "token count under max_seq_length, and whether thought markers appear."
