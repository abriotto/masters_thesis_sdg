#!/bin/bash
#SBATCH --job-name=just_full
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/just_full_%A_%a.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/just_full_%A_%a.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --account=thesis
#SBATCH --array=0-11

# Full-corpus annotation, as a job array: one task per (model, run) shard,
# 12 shards of 191 justifications each.
#
# Why an array rather than one job doing all twelve: each task gets its own
# wall clock, its own log and its own output file. A shard that hangs or
# times out is resubmitted on its own (`--array=7`) instead of taking the
# other eleven down with it.
#
# Wall time is 8h for a shard measured at ~3.6h (191 calls x 68s, from the
# pilot). The margin is deliberate -- the slowest pilot call took 152s, and a
# task that dies at the limit wastes every call it had not yet written.
#
# No --gres=gpu, 1 CPU and 2G per task, all for the same reason: this only
# makes HTTPS calls and spends ~99% of its runtime blocked on the network.
#
# That sizing matters on the debug partition, which has 30 CPUs across 5
# nodes at 6 CPUs/node. At 1 CPU per task the array uses 12 of 30 (packed
# ~6 per node, so ~2 nodes) and leaves the partition usable by others for
# the ~3.6h it runs. At 2 CPUs it would take 24 of 30 and touch ~4 nodes
# for no speed gain whatsoever.
#
# Submit:
#   sbatch slurm_files/justification_annotation_full.slurm
#
# If the partition limits concurrent tasks, throttle without editing this file:
#   sbatch --array=0-11%4 slurm_files/justification_annotation_full.slurm
#
# Re-run a single shard:
#   sbatch --array=7 slurm_files/justification_annotation_full.slurm

set -euo pipefail

REPO=/home/abriotto/Tesi/masters_thesis_sdg
cd "$REPO" || exit 1

mkdir -p "$REPO/logs"

SCHEMA=${SCHEMA:-v3}
FULL_DIR=results/justification_annotation/full_${SCHEMA}
MANIFEST=$FULL_DIR/manifest.json

echo "=== Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "started: $(date --iso-8601=seconds)"
echo "schema : $SCHEMA"

# ------------------------------------------------------------
# API key -- never written into this file, which is in git.
#   printf 'export DEEPSEEK_API_KEY=sk-...\n' > ~/.deepseek_env && chmod 600 ~/.deepseek_env
# ------------------------------------------------------------

if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "$HOME/.deepseek_env" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/.deepseek_env"
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set and ~/.deepseek_env was not found." >&2
    exit 1
fi

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

# conda activate <your_env>   # uncomment if the cluster needs it
PYTHON=${PYTHON:-python}

$PYTHON -c "import openai, pandas; print('openai', openai.__version__, '| pandas', pandas.__version__)"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: $MANIFEST not found. Build the shards first:" >&2
    echo "  $PYTHON src/pt_annotation/justification_build_input.py --schema $SCHEMA" >&2
    exit 1
fi

# ------------------------------------------------------------
# Resolve this task's shard from the manifest
#
# The manifest is the single source of truth for index -> shard, so the array
# index means the same thing here, in the builder, and in the merge step.
# ------------------------------------------------------------

read -r SHARD_KEY SHARD_IN SHARD_OUT SHARD_N < <(
    $PYTHON - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
index = int(sys.argv[2])
for shard in manifest["shards"]:
    if shard["index"] == index:
        print(shard["key"], shard["input"], shard["output"], shard["n_justifications"])
        break
else:
    sys.exit(f"No shard with index {index} in the manifest")
PYEOF
)

echo "shard  : [$SLURM_ARRAY_TASK_ID] $SHARD_KEY  ($SHARD_N justifications)"
echo "input  : $FULL_DIR/$SHARD_IN"
echo "output : $FULL_DIR/$SHARD_OUT"
echo

# --resume makes this idempotent: only justifications not already annotated
# successfully are sent. A resubmitted shard costs nothing for work already done.
$PYTHON src/pt_annotation/justification_annotation_deepseek.py \
    --schema "$SCHEMA" \
    --input-path "$FULL_DIR/$SHARD_IN" \
    --output-path "$FULL_DIR/$SHARD_OUT" \
    --resume

echo
echo "finished: $(date --iso-8601=seconds)"
