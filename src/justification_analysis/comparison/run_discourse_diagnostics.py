"""Qualitative examples, per-model statistics, and the validation sample."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is discovered, never hard-coded to one machine.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.justification_analysis.comparison import discourse_comparison as dc
from src.utils.sentences import count_sentences

from src.justification_analysis.pipeline.config import default_config

# Stage-aware. The validation sample this writes is drawn from the ACTIVE
# corpus; an old sample is never reused across stages.
CONFIG = default_config(repo_root=ROOT)
OUT = CONFIG.discourse_dir
pd.set_option("display.width", 210, "display.max_columns", 40, "display.max_colwidth", 95)

dim_a = pd.read_csv(OUT / "alignment_dimlex_side.csv", encoding="utf-8-sig")
dis_a = pd.read_csv(OUT / "alignment_discopy_side.csv", encoding="utf-8-sig")

def show(frame, n, cols, title):
    print("\n" + "-" * 88)
    print(title)
    print("-" * 88)
    if not len(frame):
        print("  (none)")
        return
    for r in frame.head(n).itertuples(index=False):
        d = dict(zip(cols, [getattr(r, c) for c in cols]))
        print("  " + " | ".join(f"{k}={d[k]!r}" for k in cols if k != "sentence_text"))
        print(f"      {str(d.get('sentence_text',''))[:170]}")

rng = np.random.RandomState(7)

print("=" * 88)
print("1. DiMLex LEXICAL MATCH THAT discopy REJECTS AS NON-CONNECTIVE")
print("=" * 88)
for form in ["for", "and", "as", "or"]:
    sub = dim_a[(dim_a.marker == form) & (dim_a.alignment_status == dc.CANDIDATE_REJECTED_NOSENSE)]
    if len(sub):
        show(sub.sample(min(3, len(sub)), random_state=rng),
             3, ["marker", "category", "discopy_confidence", "sentence_text"],
             f"'{form}' rejected as NoSense  (n={len(sub):,})")

print("\n" + "=" * 88)
print("2. SAME MARKER, DIFFERENT CONTEXTUAL SENSES")
print("=" * 88)
for form in ["as", "while", "since"]:
    sub = dim_a[(dim_a.marker == form) & (dim_a.alignment_status == dc.ALIGNED_CONNECTIVE)]
    if len(sub):
        print(f"\n'{form}' (DiMLex majority={sub.category.iat[0]}) contextual senses:")
        print(sub.discopy_top_level.value_counts().to_string())
        for cat in sub.discopy_top_level.dropna().unique():
            ex = sub[sub.discopy_top_level == cat].sample(1, random_state=rng)
            print(f"   -> {cat}: {str(ex.sentence_text.iat[0])[:150]}")

print("\n" + "=" * 88)
print("3. COVERAGE GAPS - never enumerated by discopy")
print("=" * 88)
for form in ["with", "given", "despite", "given that", "eventually"]:
    sub = dim_a[(dim_a.marker == form) & dim_a.is_coverage_evidence]
    if len(sub):
        ex = sub.sample(min(2, len(sub)), random_state=rng)
        print(f"\n'{form}' [{sub.category.iat[0]}] n={len(sub):,}")
        for r in ex.itertuples(index=False):
            print(f"      {str(r.sentence_text)[:165]}")

print("\n" + "=" * 88)
print("4. discopy-ONLY ACCEPTED (no DiMLex occurrence at that span)")
print("=" * 88)
donly = dis_a[dis_a.is_connective & (dis_a.alignment_status != dc.DISCOPY_ALIGNED)]
print(f"n={len(donly)}")
for r in donly.head(8).itertuples(index=False):
    print(f"  {r.candidate_surface!r:<14} {r.raw_sense:<22} p={r.confidence:.2f}")
    print(f"      {str(r.sentence_text)[:165]}")

print("\n" + "=" * 88)
print("5. CONFIDENCE OF REJECTIONS vs ACCEPTANCES")
print("=" * 88)
acc = dis_a[dis_a.is_connective]["confidence"]
rej = dis_a[~dis_a.is_connective]["confidence"]
print(f"accepted  n={len(acc):,} mean={acc.mean():.3f} median={acc.median():.3f} "
      f"pct<0.5={100*(acc<0.5).mean():.1f}%")
print(f"rejected  n={len(rej):,} mean={rej.mean():.3f} median={rej.median():.3f} "
      f"pct<0.5={100*(rej<0.5).mean():.1f}%")
print("\nrejection confidence by form (top DiMLex forms):")
r = dis_a[~dis_a.is_connective].assign(form=lambda d: d.candidate_surface.map(dc.normalise_surface))
print(r.groupby("form")["confidence"].agg(["size", "mean"]).sort_values("size", ascending=False).head(10).to_string())

# ---------------------------------------------------------------- statistics
print("\n" + "=" * 88)
print("6. PER-MODEL STATISTICS - discopy explicit relations")
print("=" * 88)

votes = pd.read_csv(OUT / "dimlex_occurrences.csv", encoding="utf-8-sig")
import re
WORD = re.compile(r"\b[\w]+(?:['\u2019\-][\w]+)*\b", re.UNICODE)

# Rebuild the per-justification frame with the MAIN NOTEBOOK's word counter.
REL = Path("base/voting/prompt_v4/vote_stability/tables/llm_vote_file_level.csv")
frames = []
for pat, name in [("*gemma-4-E2B*", "Gemma 4 2B"), ("*gemma-4-E4B*", "Gemma 4 4B"),
                  ("*gemma-4-31B*", "Gemma 4 31B")]:
    for d in (ROOT / "analysis").glob(pat):
        if (d / REL).exists():
            f = pd.read_csv(d / REL); f["model"] = name; frames.append(f)
J = pd.concat(frames, ignore_index=True)
J["justification"] = J["justification"].fillna("").astype(str)
J["run_number"] = J["run_label"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
J["decoding_group"] = np.where(J["decoding"].str.lower().eq("stochastic"), "Stochastic", "Greedy")
J["justification_id"] = np.arange(len(J))
J["n_words"] = J["justification"].map(lambda t: len(WORD.findall(t)))
J["n_sentences"] = J["justification"].map(count_sentences)
assert J["n_words"].sum() == 169748, J["n_words"].sum()  # same denominator as the notebook

acc_occ = dis_a[dis_a.is_connective].copy()
counts = acc_occ.groupby("justification_id").size().rename("n_conn")
J = J.merge(counts, left_on="justification_id", right_index=True, how="left")
J["n_conn"] = J["n_conn"].fillna(0).astype(int)
J["has_conn"] = (J["n_conn"] > 0).astype(int)

RUN_KEYS = ["model", "decoding_group", "run_label"]
run_level = J.groupby(RUN_KEYS, as_index=False).agg(
    n_just=("justification_id", "nunique"), total_words=("n_words", "sum"),
    total_conn=("n_conn", "sum"), n_with=("has_conn", "sum"))
run_level["per_just"] = run_level.total_conn / run_level.n_just
run_level["per_100w"] = 100 * run_level.total_conn / run_level.total_words
run_level["pct_with"] = 100 * run_level.n_with / run_level.n_just
assert (run_level.n_just == 191).all()

summary = run_level.groupby(["model", "decoding_group"]).agg(
    n_runs=("run_label", "nunique"),
    per_just_mean=("per_just", "mean"), per_just_sd=("per_just", "std"),
    per_100w_mean=("per_100w", "mean"), per_100w_sd=("per_100w", "std"),
    pct_with_mean=("pct_with", "mean"), pct_with_sd=("pct_with", "std"))
print(summary.round(3).to_string())

print("\nCategory rate per 100 words (discopy), by model x decoding:")
cat_rows = []
for cat in dc.PDTB_TOP_LEVEL:
    c = acc_occ[acc_occ.top_level == cat].groupby("justification_id").size().rename("n")
    tmp = J[["justification_id", "model", "decoding_group", "run_label", "n_words"]].merge(
        c, left_on="justification_id", right_index=True, how="left")
    tmp["n"] = tmp["n"].fillna(0)
    rl = tmp.groupby(RUN_KEYS).apply(lambda g: 100 * g["n"].sum() / g["n_words"].sum())
    s = rl.groupby(["model", "decoding_group"]).agg(["mean", "std"])
    s.columns = [f"{cat}_{c}" for c in s.columns]
    cat_rows.append(s)
cat_tab = pd.concat(cat_rows, axis=1)
print(cat_tab.round(3).to_string())

print("\nCategory PROPORTIONS of accepted connectives, by model:")
prop = (acc_occ.merge(J[["justification_id", "model"]], on="justification_id", suffixes=("", "_j"))
        .groupby(["model_j", "top_level"]).size().unstack(fill_value=0))
print((100 * prop.div(prop.sum(axis=1), axis=0)).round(1).to_string())

run_level.to_csv(OUT / "discopy_run_level_statistics.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUT / "discopy_model_summary.csv", encoding="utf-8-sig")
cat_tab.to_csv(OUT / "discopy_category_rates_per100w.csv", encoding="utf-8-sig")
print(f"\nsaved statistics -> {OUT}")
