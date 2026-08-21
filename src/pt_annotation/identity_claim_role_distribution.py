"""
Role-distribution statistics over the identity-claim annotation output.

Reads every per-game JSON under ic_targets/, normalises the free-text role
labels, and reports the corpus-level counts used in the validation subsection
(plus the LaTeX table body).

Normalisation is applied at analysis time rather than by rewriting the
annotation files, so the stored annotations stay exactly as produced (by the
model, or by the manual review notebook) and every derived number in the thesis
is reproducible from this one script.

Three things need normalising:

1. Case. The model writes Title Case ("Troublemaker"); the manual review
   notebook wrote its resolutions lower case ("troublemaker"). Same role.
2. Diacritics. Both "Doppelganger" and "Doppelgaenger"/"Doppelgänger" occur.
   Same role.
3. One-off labels. "Double Werewolf" is a Werewolf claim ("May or may not be a
   werewolf or a villager. I'm double werewolf.") and is folded into Werewolf.

Roles outside the twelve described in the rules prompt (onuw_rules_v2.txt) come
from expansions used in a handful of recorded games. They are reported together
in a single "Other" bin rather than as long-tail rows, and listed individually
in the run summary so the composition of that bin stays inspectable.

Usage:
    python src/pt_annotation/identity_claim_role_distribution.py
    python src/pt_annotation/identity_claim_role_distribution.py --latex
"""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT_ROOT = Path(
    r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg\data\processed"
    r"\lai2023\identity_claim_transcripts\ic_targets"
)

# Sentinel the annotator uses when a claim was made but the role is unresolved.
UNKNOWN_ROLE = "unknown"

# Folded label -> canonical display name. Anything not listed falls back to
# title case, which is already correct for every standard role name.
ROLE_ALIASES = {
    "doppelganger": "Doppelgänger",
    "double werewolf": "Werewolf",
}

# The twelve roles described in src/prompts/onuw_rules_v2.txt. Everything else
# is an expansion role and is aggregated into OTHER_LABEL.
CANONICAL_ROLES = (
    "Doppelgänger",
    "Drunk",
    "Hunter",
    "Insomniac",
    "Mason",
    "Minion",
    "Robber",
    "Seer",
    "Tanner",
    "Troublemaker",
    "Villager",
    "Werewolf",
)

OTHER_LABEL = "Other"


# ============================================================
# Normalisation
# ============================================================

def fold_label(raw):
    """Case-, whitespace- and diacritic-insensitive key for a role label."""
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.split()).casefold()


def normalise_role(raw):
    """Display name for a role label, or None if the label is empty."""
    key = fold_label(raw)
    if not key:
        return None
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    return key.title()


# ============================================================
# Aggregation
# ============================================================

def collect_stats(input_root):
    counts = Counter()
    other_breakdown = Counter()  # what actually landed in the "Other" bin
    totals = {
        "files": 0,
        "items": 0,
        "empty": 0,          # marked line, but no role actually claimed
        "unknown": 0,        # claim made, role still unresolved
        "resolved": 0,       # at least one concrete role claimed
        "multi_role": 0,     # resolved items claiming more than one role
    }

    for json_path in sorted(input_root.rglob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        totals["files"] += 1

        for item in record.get("items", []):
            totals["items"] += 1
            claimed = item.get("claimed_roles") or []

            if not claimed:
                totals["empty"] += 1
                continue
            if any(fold_label(r) == UNKNOWN_ROLE for r in claimed):
                totals["unknown"] += 1
                continue

            totals["resolved"] += 1
            if len(claimed) > 1:
                totals["multi_role"] += 1
            for raw in claimed:
                role = normalise_role(raw)
                if role is None:
                    continue
                if role in CANONICAL_ROLES:
                    counts[role] += 1
                else:
                    counts[OTHER_LABEL] += 1
                    other_breakdown[role] += 1

    return counts, totals, other_breakdown


def ordered_rows(counts):
    """Counts sorted by frequency, with the Other bin pinned to the bottom."""
    rows = [(role, n) for role, n in counts.most_common() if role != OTHER_LABEL]
    if counts[OTHER_LABEL]:
        rows.append((OTHER_LABEL, counts[OTHER_LABEL]))
    return rows


# ============================================================
# Reporting
# ============================================================

def print_summary(counts, totals, other_breakdown):
    rows = ordered_rows(counts)
    mentions = sum(counts.values())
    items = totals["items"] or 1

    print(f"transcripts            : {totals['files']}")
    print(f"identity-claim items   : {totals['items']}")
    print(f"  resolved to role(s)  : {totals['resolved']} ({100 * totals['resolved'] / items:.1f}%)")
    print(f"  no role claimed      : {totals['empty']} ({100 * totals['empty'] / items:.1f}%)")
    print(f"  unresolved (UNKNOWN) : {totals['unknown']} ({100 * totals['unknown'] / items:.1f}%)")
    print(f"  claiming >1 role     : {totals['multi_role']}")
    print(f"role mentions counted  : {mentions}")
    print()

    canonical_rows = [(role, n) for role, n in rows if role != OTHER_LABEL]
    top_five = sum(count for _, count in canonical_rows[:5])
    print(f"top-5 share            : {100 * top_five / mentions:.1f}%")
    print()

    for role, count in rows:
        print(f"  {role:<16} {count:>5}  {100 * count / mentions:>5.1f}%")

    if other_breakdown:
        print()
        print(f"  '{OTHER_LABEL}' comprises (expansion roles, not in onuw_rules_v2.txt):")
        for role, count in other_breakdown.most_common():
            print(f"    {role:<16} {count:>3}")


def print_latex(counts):
    rows = ordered_rows(counts)
    mentions = sum(counts.values())
    print(r"\begin{tabular}{lrr}")
    print(r"  \toprule")
    print(r"  Claimed role & Mentions & \% \\")
    print(r"  \midrule")
    for role, count in rows:
        label = role.replace("ä", r"\"a")
        print(f"  {label:<18} & {count:>5} & {100 * count / mentions:>5.1f} \\\\")
    print(r"  \midrule")
    print(f"  {'Total':<18} & {mentions:>5} & {100 * mentions / mentions:>5.1f} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(
        description="Role-distribution statistics over identity-claim annotations."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--latex", action="store_true", help="Emit the LaTeX table body instead of the summary.")
    args = parser.parse_args()

    counts, totals, other_breakdown = collect_stats(args.input_root)
    if args.latex:
        print_latex(counts)
    else:
        print_summary(counts, totals, other_breakdown)


if __name__ == "__main__":
    main()
