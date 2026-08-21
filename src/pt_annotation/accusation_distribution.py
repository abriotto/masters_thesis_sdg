"""
Distribution statistics over the accusation annotation output.

Reads every per-game JSON under acc_targets/ and reports the corpus-level counts
used in the validation and distribution subsections, plus the LaTeX table body.

Counterpart to identity_claim_role_distribution.py. As there, everything is
computed at analysis time from the stored annotations, so every number in the
thesis is reproducible from this one script.

Units of counting, which should not be conflated:

* an *instance* is one utterance carrying the upstream Accusation label (one
  marked line, one item in the JSON);
* an *accusation* is one (subtype, single target) pair. A stored relation naming
  several players jointly -- "you two are the werewolves" -- therefore expands to
  one accusation per player named, so that every accusation has exactly one
  accuser and exactly one target.

An instance may yield no accusation at all -- the upstream label is broader than
the two subtypes annotated here -- or several, either because the speaker makes
distinct accusations in one turn or because a single accusation names several
players.

Only accusations that are of a relevant subtype *and* carry a resolved target are
counted: relations left as UNKNOWN are dropped, and an instance whose only
relation was unresolved leaves the analysis set with them. The raw counts are
still reported so the validation section can state what was dropped.

Usage:
    python src/pt_annotation/accusation_distribution.py
    python src/pt_annotation/accusation_distribution.py --latex
"""

import argparse
import json
from collections import Counter
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT_ROOT = Path(
    r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg\data\processed"
    r"\lai2023\accusation_transcripts\acc_targets"
)

UNKNOWN_TARGET = "UNKNOWN"

# Display order for the two subtypes, most frequent first.
TYPE_ORDER = ("werewolf", "deception")

TYPE_LABELS = {
    "werewolf": "Werewolf attribution",
    "deception": "Deception",
}


# ============================================================
# Aggregation
# ============================================================

def collect_stats(input_root):
    totals = {
        "files": 0,
        "instances": 0,         # utterances carrying the upstream Accusation label
        "no_relation": 0,       # ... yielding neither subtype
        "with_relation": 0,     # ... yielding at least one relation, resolved or not
        "relations": 0,         # stored relations, resolved or not
        "unknown_relations": 0, # relations whose target stayed UNKNOWN
        "unknown_instances": 0, # instances holding at least one such relation
        "instances_lost": 0,    # instances leaving the analysis set entirely
        # --- analysis set ---
        "kept_instances": 0,
        "kept_relations": 0,    # resolved relations, before expanding group targets
        "accusations": 0,       # one per (subtype, single target)
        "from_group": 0,        # accusations coming from a jointly-named group
    }
    types = Counter()                  # subtype counts, per accusation
    targets_per_relation = Counter()   # how many players a resolved relation names
    accusations_per_instance = Counter()
    duplicates = 0                     # same subtype+target twice in one instance

    for json_path in sorted(input_root.rglob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        totals["files"] += 1

        for item in record.get("items", []):
            totals["instances"] += 1
            relations = item.get("relations") or []

            if not relations:
                totals["no_relation"] += 1
                continue
            totals["with_relation"] += 1

            kept = []
            has_unknown = False
            for relation in relations:
                totals["relations"] += 1
                if UNKNOWN_TARGET in (relation.get("accused") or []):
                    totals["unknown_relations"] += 1
                    has_unknown = True
                else:
                    kept.append(relation)
            if has_unknown:
                totals["unknown_instances"] += 1

            if not kept:
                totals["instances_lost"] += 1
                continue

            totals["kept_instances"] += 1
            totals["kept_relations"] += len(kept)

            seen = set()
            n_here = 0
            for relation in kept:
                accused = relation.get("accused") or []
                subtype = relation.get("type")
                targets_per_relation[len(accused)] += 1
                for target in accused:
                    totals["accusations"] += 1
                    n_here += 1
                    types[subtype] += 1
                    if len(accused) > 1:
                        totals["from_group"] += 1
                    if (subtype, target) in seen:
                        duplicates += 1
                    seen.add((subtype, target))
            accusations_per_instance[n_here] += 1

    return totals, types, targets_per_relation, accusations_per_instance, duplicates


# ============================================================
# Reporting
# ============================================================

def print_summary(totals, types, targets_per_relation, accusations_per_instance, duplicates):
    instances = totals["instances"] or 1
    accusations = totals["accusations"] or 1

    print("--- all output ---")
    print(f"transcripts                  : {totals['files']}")
    print(f"Accusation instances         : {totals['instances']}")
    print(f"  yielding >=1 relation      : {totals['with_relation']} ({100 * totals['with_relation'] / instances:.1f}%)")
    print(f"  yielding no relation       : {totals['no_relation']} ({100 * totals['no_relation'] / instances:.1f}%)")
    print(f"stored relations             : {totals['relations']}")
    print(f"  unresolved target          : {totals['unknown_relations']} "
          f"(in {totals['unknown_instances']} instances)")
    print(f"  instances dropped entirely : {totals['instances_lost']}")
    print()
    print("--- analysis set (relevant subtype AND resolved target) ---")
    print(f"instances                    : {totals['kept_instances']}")
    print(f"resolved relations           : {totals['kept_relations']}")
    print(f"accusations (one per target) : {totals['accusations']}")
    print(f"  from a single-target relation : {totals['accusations'] - totals['from_group']}")
    print(f"  from a jointly-named group    : {totals['from_group']}")
    if duplicates:
        print(f"  NOTE: {duplicates} duplicate (subtype, target) pairs within one instance")
    print()
    for name in TYPE_ORDER:
        print(f"  {name:<26} {types[name]} ({100 * types[name] / accusations:.1f}%)")
    other = {t: n for t, n in types.items() if t not in TYPE_ORDER}
    for name, count in other.items():
        print(f"  {str(name):<26} {count}  <-- unexpected type")
    print()
    print("accusations per instance:")
    for size in sorted(accusations_per_instance):
        print(f"  {size:>2} : {accusations_per_instance[size]}")
    print("players named per stored relation:")
    for size in sorted(targets_per_relation):
        print(f"  {size:>2} : {targets_per_relation[size]}")


def print_latex(totals, types):
    accusations = totals["accusations"] or 1
    print(r"\begin{tabular}{lrr}")
    print(r"  \toprule")
    print(r"  \textbf{Accusation subtype} & \textbf{Accusations} & \textbf{\%} \\")
    print(r"  \midrule")
    for name in TYPE_ORDER:
        label = TYPE_LABELS.get(name, str(name).title())
        print(f"  {label:<22} & {types[name]:>5} & {100 * types[name] / accusations:>5.1f} \\\\")
    print(r"  \midrule")
    print(f"  {'Total':<22} & {totals['accusations']:>5} & 100.0 \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(
        description="Distribution statistics over accusation annotations."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--latex", action="store_true", help="Emit the LaTeX table body instead of the summary.")
    args = parser.parse_args()

    totals, types, targets, per_instance, duplicates = collect_stats(args.input_root)
    if args.latex:
        print_latex(totals, types)
    else:
        print_summary(totals, types, targets, per_instance, duplicates)


if __name__ == "__main__":
    main()
