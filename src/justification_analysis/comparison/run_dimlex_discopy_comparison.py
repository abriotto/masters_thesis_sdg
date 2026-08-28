"""Run the DiMLex vs discopy comparison and write the diagnostic tables."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg")
sys.path.insert(0, str(ROOT))
SP = Path(__file__).resolve().parent.parent / "discopy_parser"

from src.justification_analysis.comparison import discourse_comparison as dc

OUT = ROOT / "analysis/cross_model/base/voting/prompt_v4/justification_analysis/discourse_parser"
pd.set_option("display.width", 200, "display.max_columns", 40)

dimlex = pd.read_csv(OUT / "dimlex_occurrences.csv", encoding="utf-8-sig")
dimlex["span_list"] = dimlex["char_spans"].map(dc.parse_char_spans)
discopy = dc.load_discopy_candidates(OUT / "discopy_explicit_candidates.csv")

inventory = json.load(open(SP / "discopy_connectives.json"))
inv_forms = inventory["single"] + inventory["multi"] + inventory["distant"]

print("=" * 90)
print("HEADLINE COUNTS")
print("=" * 90)
print(f"DiMLex lexical occurrences        : {len(dimlex):,}")
print(f"discopy candidates enumerated     : {len(discopy):,}")
print(f"discopy accepted as connectives   : {int(discopy['is_connective'].sum()):,}")
print(f"discopy rejected (NoSense)        : {int((~discopy['is_connective']).sum()):,}")
print(f"discopy rejection rate            : {100*(~discopy['is_connective']).mean():.1f}%")

dim_a, dis_a = dc.align_dimlex_discopy_occurrences(dimlex, discopy, inv_forms)

print("\n" + "=" * 90)
print("COVERAGE vs CLASSIFICATION  (every DiMLex occurrence)")
print("=" * 90)
summary = dc.coverage_vs_classification_summary(dim_a)
print(summary.to_string(index=False))

cov = int(dim_a["is_coverage_evidence"].sum())
cls = int(dim_a["is_classification_evidence"].sum())
print(f"\n  candidate-coverage evidence (hybrid WOULD fix) : {cov:,}")
print(f"  contextual-classification evidence (would NOT) : {cls:,}")

print("\n" + "=" * 90)
print("discopy SIDE")
print("=" * 90)
print(dis_a.loc[dis_a["is_connective"], "alignment_status"]
      .value_counts().to_string())

print("\n" + "=" * 90)
print("SENSE CHANGE CROSSTAB (aligned connectives only)")
print("=" * 90)
ct = dc.sense_change_crosstab(dim_a)
print(ct.to_string())
both = dim_a.loc[dim_a["alignment_status"] == dc.ALIGNED_CONNECTIVE]
same = int((~both["category_changed"].astype(bool)).sum())
print(f"\naligned occurrences: {len(both):,}")
print(f"  same top-level category   : {same:,} ({100*same/len(both):.1f}%)")
print(f"  category changed          : {len(both)-same:,} ({100*(len(both)-same)/len(both):.1f}%)")

print("\n" + "=" * 90)
print("PER-FORM INVENTORY COMPARISON (top 25 by DiMLex count)")
print("=" * 90)
inv = dc.compare_connective_inventories(dim_a, dis_a)
cols = ["dimlex_category", "n_dimlex", "n_aligned", "n_rejected_nosense",
        "n_outside_inventory", "n_in_inventory_not_enumerated",
        "n_discopy_accepted", "n_discopy_only", "n_category_changed",
        "pct_dimlex_retained"]
print(inv[cols].head(25).to_string())

print("\n" + "=" * 90)
print("THE FOUR AMBIGUOUS FORMS OF INTEREST")
print("=" * 90)
print(inv.loc[inv.index.isin(["as", "and", "for", "with"]), cols].to_string())

print("\n" + "=" * 90)
print("LESS AMBIGUOUS CONNECTIVES")
print("=" * 90)
print(inv.loc[inv.index.isin(
    ["because", "but", "however", "therefore", "although", "if then",
     "either or", "since", "while", "so"]), cols].to_string())

print("\n" + "=" * 90)
print("TOP COVERAGE GAPS (never enumerated by discopy)")
print("=" * 90)
gaps = inv.loc[inv["n_coverage_gap"] > 0].sort_values("n_coverage_gap", ascending=False)
print(gaps[["dimlex_category", "n_dimlex", "n_coverage_gap",
            "n_outside_inventory", "n_in_inventory_not_enumerated"]].head(25).to_string())
print(f"\ntotal coverage-gap occurrences: {int(inv['n_coverage_gap'].sum()):,}")
print(f"  form outside discopy inventory        : {int(inv['n_outside_inventory'].sum()):,}")
print(f"  form in inventory, not enumerated here: {int(inv['n_in_inventory_not_enumerated'].sum()):,}")

print("\n" + "=" * 90)
print("discopy-ONLY ACCEPTED CONNECTIVES")
print("=" * 90)
donly = dis_a.loc[dis_a["is_connective"] & (dis_a["alignment_status"] != dc.DISCOPY_ALIGNED)]
print(f"total: {len(donly):,}")
print(donly["alignment_status"].value_counts().to_string())
print("\nby form:")
print(donly.assign(form=donly["candidate_surface"].map(dc.normalise_surface))
      .groupby("form").size().sort_values(ascending=False).head(20).to_string())

dim_a.drop(columns=["span_list"]).to_csv(OUT / "alignment_dimlex_side.csv", index=False, encoding="utf-8-sig")
dis_a.drop(columns=["span_list"]).to_csv(OUT / "alignment_discopy_side.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUT / "coverage_vs_classification_summary.csv", index=False, encoding="utf-8-sig")
ct.to_csv(OUT / "sense_change_crosstab.csv", encoding="utf-8-sig")
inv.to_csv(OUT / "connective_inventory_comparison.csv", encoding="utf-8-sig")
print(f"\nsaved alignment + diagnostic tables -> {OUT}")
