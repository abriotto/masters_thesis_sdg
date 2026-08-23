"""
Status and integrity check for the sharded full-corpus annotation run.

Answers, in one command: is every shard finished, did anything fail, and is
the output actually usable. With twelve array tasks writing twelve files,
"the job finished" and "the corpus is annotated" are not the same claim --
a task can exit 0 having errored on individual justifications, and a shard
can be short without anything looking broken.

Exit code is 0 only when every shard is complete and unflagged, so this can
gate the merge step in a script.

Usage:
    python src/pt_annotation/justification_check_run.py
    python src/pt_annotation/justification_check_run.py --schema v3
    python src/pt_annotation/justification_check_run.py --merge
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pt_annotation.justification_schema import DEFAULT_SCHEMA, SCHEMAS, get_schema  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results" / "justification_annotation"


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def inspect_shard(full_dir, shard):
    output_path = full_dir / shard["output"]
    expected = shard["n_justifications"]

    state = {
        "index": shard["index"],
        "key": shard["key"],
        "expected": expected,
        "annotated": 0,
        "errored": 0,
        "flagged": 0,
        "missing": expected,
        "duplicates": 0,
        "exists": output_path.exists(),
    }

    if not output_path.exists():
        return state

    records = [r for r in read_jsonl(output_path) if not r["metadata"].get("dry_run")]

    ids = [r["metadata"]["justification_id"] for r in records]
    counts = Counter(ids)
    state["duplicates"] = sum(1 for _, n in counts.items() if n > 1)

    # A resumed shard appends, so an id can legitimately appear twice: once
    # errored, once succeeded. Only successful records count as done.
    done = {
        r["metadata"]["justification_id"]
        for r in records
        if r.get("annotation") is not None and "error" not in r["metadata"]
    }
    state["annotated"] = len(done)
    state["missing"] = expected - len(done)
    state["errored"] = sum(1 for r in records if "error" in r["metadata"])
    state["flagged"] = sum(1 for r in records if r["metadata"].get("validation_flags"))
    state["records"] = records

    return state


def main():
    parser = argparse.ArgumentParser(description="Check the sharded annotation run.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, choices=sorted(SCHEMAS))
    parser.add_argument("--full-dir", type=Path, default=None)
    parser.add_argument(
        "--merge", action="store_true",
        help="Write annotations.jsonl combining every shard (only if all are complete).",
    )
    args = parser.parse_args()

    full_dir = args.full_dir or (RESULTS_ROOT / f"full_{args.schema}")
    manifest_path = full_dir / "manifest.json"

    if not manifest_path.exists():
        sys.exit(f"No manifest at {manifest_path}. Run justification_build_input.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = get_schema(manifest.get("schema", args.schema))

    states = [inspect_shard(full_dir, shard) for shard in manifest["shards"]]

    print(f"{full_dir.relative_to(REPO_ROOT)}  (schema {schema.name})\n")
    print(f"{'idx':>3}  {'shard':<16} {'done':>10} {'err':>5} {'flag':>5}  status")
    print("-" * 62)

    for state in states:
        if not state["exists"]:
            status = "NOT STARTED"
        elif state["missing"] > 0:
            status = f"INCOMPLETE ({state['missing']} left)"
        elif state["errored"] or state["flagged"]:
            status = "done, needs review"
        else:
            status = "complete"

        print(
            f"{state['index']:>3}  {state['key']:<16} "
            f"{state['annotated']:>4}/{state['expected']:<5} "
            f"{state['errored']:>5} {state['flagged']:>5}  {status}"
        )

    total_expected = sum(s["expected"] for s in states)
    total_done = sum(s["annotated"] for s in states)
    total_errored = sum(s["errored"] for s in states)
    total_flagged = sum(s["flagged"] for s in states)

    print("-" * 62)
    print(f"     {'TOTAL':<16} {total_done:>4}/{total_expected:<5} "
          f"{total_errored:>5} {total_flagged:>5}")

    incomplete = [s for s in states if s["missing"] > 0 or not s["exists"]]

    if incomplete:
        indices = ",".join(str(s["index"]) for s in incomplete)
        print(f"\n{len(incomplete)} shard(s) incomplete. Resubmit just those:")
        print(f"  sbatch --array={indices} slurm_files/justification_annotation_full.slurm")
        print("  (--resume means finished justifications are not re-paid for)")

    if total_flagged:
        print(f"\n{total_flagged} justification(s) carry validation flags. Sample:")
        shown = 0
        for state in states:
            for record in state.get("records", []):
                flags = record["metadata"].get("validation_flags")
                if flags and shown < 5:
                    print(f"  {record['metadata']['justification_id']}")
                    for flag in flags[:2]:
                        print(f"    - {flag}")
                    shown += 1

    if args.merge:
        if incomplete:
            sys.exit("\nRefusing to merge: not every shard is complete.")

        merged_path = full_dir / "annotations.jsonl"
        seen = set()
        written = 0
        with merged_path.open("w", encoding="utf-8") as handle:
            for state in states:
                for record in state.get("records", []):
                    justification_id = record["metadata"]["justification_id"]
                    if record.get("annotation") is None or "error" in record["metadata"]:
                        continue
                    if justification_id in seen:
                        continue          # keep the first success, drop resume duplicates
                    seen.add(justification_id)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
        print(f"\nMerged {written} annotations -> {merged_path.relative_to(REPO_ROOT)}")

    sys.exit(0 if not incomplete and not total_errored else 1)


if __name__ == "__main__":
    main()
