"""
BASE -> FT contrast for the RQ3 voting subsection.

Notebook 2 compares models within one stage; it has no path to a direct
BASE -> finetuned contrast, so that contrast is computed here. Every
definition is copied unchanged from notebooks/03_llm_voting_outcome_analysis:
p_correct and the vote distribution from notebook 1, and bootstrap_mean_ci,
within/between_model_agreement, the availability-normalised selection lift and
the start/end Werewolf split from notebook 2. Only the pairing is new -- each
model is compared with its own finetuned counterpart on the same games, and
the per-game difference is bootstrapped.

Reads the tables notebook 1 writes under analysis/<model>/<stage>/, so run
notebook 1 for the finetuned dirs first:

    EXPECT_GREEDY_RUN=0 LLM_NAME=<results/voting dir> jupyter nbconvert         --execute --to notebook 1_llm_vote_tables.ipynb

Run:  python -m src.voting.ft_vote_contrast
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import json
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis"
PROMPT_DIR = "prompt_v4"
CIRCLE = "No Werewolf"
N_BOOT, BOOT_SEED = 5000, 0

MODELS = {
    "E2B": ("unsloth_gemma-4-E2B-it-unsloth-bnb-4bit", "ft_gemma-4-E2B-role-inference-traced-final_adapter"),
    "E4B": ("unsloth_gemma-4-E4B-it-unsloth-bnb-4bit", "ft_gemma-4-E4B-role-inference-traced-final_adapter"),
    "31B": ("unsloth_gemma-4-31B-it-unsloth-bnb-4bit", "ft_gemma-4-31B-role-inference-traced-final_adapter"),
}
EXCLUDE = {"E2B": {"Youtube / The#Return#of#the#King##ONE#NIGHT#ULTIMATE#WEREWOLF / Game5"}}


def bootstrap_mean_ci(values, n_boot=N_BOOT, alpha=0.05, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def parse_json(value, default):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def tables(model_base, stage):
    d = ANALYSIS / model_base / stage / "voting" / PROMPT_DIR / "vote_stability" / "tables"
    g = pd.read_csv(d / "llm_vote_game_level.csv")
    for col, default in [("player_names", []), ("start_roles", []), ("end_roles", []),
                         ("correct_player_names", []), ("stoch_vote_distribution", {})]:
        g[col] = g[col].apply(lambda v: parse_json(v, default))
    g["wolves"] = g["correct_player_names"].apply(set)
    f = pd.read_csv(d / "llm_vote_file_level.csv")
    return g, f


def chance_p_correct(row):
    n_options = len(row["player_names"]) + 1
    n_ww = len(row["wolves"])
    return (n_ww / n_options) if n_ww else (1.0 / n_options)


def circle_mass(dist):
    return float(dist.get(CIRCLE, 0.0))


def within_model_agreement(labels):
    pairs = list(combinations(range(len(labels)), 2))
    return sum(labels[i] == labels[j] for i, j in pairs) / len(pairs) if pairs else np.nan


def between_model_agreement(la, lb):
    if not la or not lb:
        return np.nan
    return sum(a == b for a in la for b in lb) / (len(la) * len(lb))


def stoch_labels(f):
    s = f[(f["decoding"] == "stochastic") & f["status"].isin(["player_vote", "circle_vote"])].copy()
    s["vote"] = np.where(s["is_circle_vote"].astype(bool), CIRCLE, s["chosen_player_name"])
    return s.groupby("game_id")["vote"].apply(list).to_dict()


DATA = {}
for label, (mb, ft_stage) in MODELS.items():
    gb, fb = tables(mb, "base")
    gf, ff = tables(mb, ft_stage)
    common = (set(gb["game_id"]) & set(gf["game_id"])) - EXCLUDE.get(label, set())
    DATA[label] = {
        "base_g": gb[gb["game_id"].isin(common)].set_index("game_id").sort_index(),
        "ft_g":   gf[gf["game_id"].isin(common)].set_index("game_id").sort_index(),
        "base_l": stoch_labels(fb), "ft_l": stoch_labels(ff),
        "games": sorted(common),
    }
    for k in ("base_g", "ft_g"):
        DATA[label][k]["chance"] = DATA[label][k].apply(chance_p_correct, axis=1)
        DATA[label][k]["circle_mass"] = DATA[label][k]["stoch_vote_distribution"].apply(circle_mass)

print("=" * 92)
print("A. MATCHED GAME SETS AND VOTE COMPLETENESS")
print("=" * 92)
for label in MODELS:
    d = DATA[label]
    nb = d["base_g"]["n_valid_stochastic"]
    nf = d["ft_g"]["n_valid_stochastic"]
    print(f"{label}: n_games={len(d['games'])} | BASE draws {int(nb.sum())} "
          f"(games with <3: {int((nb < 3).sum())}) | FT draws {int(nf.sum())} "
          f"(games with <3: {int((nf < 3).sum())})")
    short = d["ft_g"][d["ft_g"]["n_valid_stochastic"] < 3]
    for gid, r in short.iterrows():
        print(f"     FT incomplete: {gid}  ({int(r['n_valid_stochastic'])}/3 valid)")

print()
print("=" * 92)
print("B. OVERALL VOTING PERFORMANCE  (paired bootstrap on FT - BASE, game resampled)")
print("=" * 92)
rows = []
for label in MODELS:
    d = DATA[label]
    b, f = d["base_g"], d["ft_g"]
    assert (b.index == f.index).all()
    diff, lo, hi = bootstrap_mean_ci((f["p_correct"] - b["p_correct"]).values)
    rows.append({"model": label, "n_games": len(b),
                 "base_acc": b["p_correct"].mean(), "ft_acc": f["p_correct"].mean(),
                 "delta": diff, "ci_low": lo, "ci_high": hi,
                 "sig": bool(lo > 0 or hi < 0),
                 "chance": b["chance"].mean()})
overall = pd.DataFrame(rows)
print("\nOverall stochastic accuracy (mean p_correct):")
print(overall.round(4).to_string(index=False))

print("\nEdge over the random-vote baseline, per condition:")
er = []
for label in MODELS:
    d = DATA[label]
    for cond, g in [("BASE", d["base_g"]), ("FT", d["ft_g"])]:
        e, lo, hi = bootstrap_mean_ci((g["p_correct"] - g["chance"]).values)
        er.append({"model": label, "cond": cond, "acc": g["p_correct"].mean(),
                   "chance": g["chance"].mean(), "edge": e, "ci_low": lo, "ci_high": hi,
                   "above_chance": bool(lo > 0)})
print(pd.DataFrame(er).round(4).to_string(index=False))

print("\nAccuracy split by whether the game contains an end-Werewolf:")
sr = []
for label in MODELS:
    d = DATA[label]
    hw = d["base_g"]["has_werewolf"].astype(bool)
    for flag, name in [(False, "no_werewolf"), (True, "werewolf_present")]:
        m = hw == flag
        b, f = d["base_g"][m], d["ft_g"][m]
        diff, lo, hi = bootstrap_mean_ci((f["p_correct"] - b["p_correct"]).values)
        sr.append({"model": label, "subset": name, "n": int(m.sum()),
                   "base_acc": b["p_correct"].mean(), "ft_acc": f["p_correct"].mean(),
                   "delta": diff, "ci_low": lo, "ci_high": hi,
                   "sig": bool(lo > 0 or hi < 0)})
print(pd.DataFrame(sr).round(4).to_string(index=False))

print("\n'No Werewolf' vote mass (share of stochastic vote mass on the circle label):")
cr = []
for label in MODELS:
    d = DATA[label]
    b, f = d["base_g"], d["ft_g"]
    diff, lo, hi = bootstrap_mean_ci((f["circle_mass"] - b["circle_mass"]).values)
    cr.append({"model": label, "base_circle": b["circle_mass"].mean(),
               "ft_circle": f["circle_mass"].mean(), "delta": diff,
               "ci_low": lo, "ci_high": hi, "sig": bool(lo > 0 or hi < 0)})
    for flag, name in [(False, " (no-ww games)"), (True, " (ww games)")]:
        m = b["has_werewolf"].astype(bool) == flag
        dd, l2, h2 = bootstrap_mean_ci((f.loc[m, "circle_mass"] - b.loc[m, "circle_mass"]).values)
        cr.append({"model": label + name, "base_circle": b.loc[m, "circle_mass"].mean(),
                   "ft_circle": f.loc[m, "circle_mass"].mean(), "delta": dd,
                   "ci_low": l2, "ci_high": h2, "sig": bool(l2 > 0 or h2 < 0)})
print(pd.DataFrame(cr).round(4).to_string(index=False))


print("=" * 92)
print("C. VOTE AGREEMENT")
print("=" * 92)
print("\nWithin-model stability (mean pairwise agreement, distinct run pairs), paired BASE->FT:")
rows = []
for label in MODELS:
    d = DATA[label]
    gb = np.array([within_model_agreement(d["base_l"][g]) for g in d["games"]], float)
    gf = np.array([within_model_agreement(d["ft_l"][g]) for g in d["games"]], float)
    diff, lo, hi = bootstrap_mean_ci(gf - gb)
    rows.append({"model": label, "n_games": len(d["games"]),
                 "base_stability": np.nanmean(gb), "ft_stability": np.nanmean(gf),
                 "delta": diff, "ci_low": lo, "ci_high": hi, "sig": bool(lo > 0 or hi < 0)})
print(pd.DataFrame(rows).round(4).to_string(index=False))

common_all = sorted(set.intersection(*[set(DATA[m]["games"]) for m in MODELS]))
print(f"\nCross-model agreement on the {len(common_all)} games common to all three models.")
order = list(MODELS)
mats = {}
for cond, key in [("BASE", "base_l"), ("FT", "ft_l")]:
    mat = pd.DataFrame(np.nan, index=order, columns=order)
    for m in order:
        mat.loc[m, m] = np.nanmean([within_model_agreement(DATA[m][key][g]) for g in common_all])
    for a, b in combinations(order, 2):
        v = np.nanmean([between_model_agreement(DATA[a][key][g], DATA[b][key][g]) for g in common_all])
        mat.loc[a, b] = mat.loc[b, a] = v
    mats[cond] = mat
    print(f"\n{cond} mean pairwise vote agreement, % (diagonal = within-model stability):")
    print((mat * 100).round(1).to_string())

print("\nFT - BASE change in agreement, percentage points:")
print(((mats["FT"] - mats["BASE"]) * 100).round(1).to_string())

print("\nPaired bootstrap per cross-model pair (FT - BASE), games resampled:")
rows = []
for a, b in combinations(order, 2):
    vb = np.array([between_model_agreement(DATA[a]["base_l"][g], DATA[b]["base_l"][g]) for g in common_all], float)
    vf = np.array([between_model_agreement(DATA[a]["ft_l"][g], DATA[b]["ft_l"][g]) for g in common_all], float)
    diff, lo, hi = bootstrap_mean_ci(vf - vb)
    rows.append({"pair": f"{a}-{b}", "base": np.nanmean(vb), "ft": np.nanmean(vf),
                 "delta": diff, "ci_low": lo, "ci_high": hi, "sig": bool(lo > 0 or hi < 0)})
print(pd.DataFrame(rows).round(4).to_string(index=False))

print()
print("=" * 92)
print("D1. INCORRECT VOTE TARGETS - availability-normalised selection lift")
print("=" * 92)

MIN_OPPORTUNITY_SHARE = 0.04
LIFT_N_BOOT, LIFT_BOOT_SEED = 5000, 20260822


def lift_matrices(g, game_ids, roles):
    """Per-game observed / expected wrong-vote mass per role (notebook 2, cell 24)."""
    gp = {gid: i for i, gid in enumerate(game_ids)}
    rp = {r: i for i, r in enumerate(roles)}
    obs = np.zeros((len(roles), len(game_ids)))
    exp = np.zeros((len(roles), len(game_ids)))
    for gid, r in g.iterrows():
        if gid not in gp:
            continue
        dist, names, ends = r["stoch_vote_distribution"], r["player_names"], r["end_roles"]
        nonwolf = [role for role in ends if role != "Werewolf"]
        if not dist or not nonwolf:
            continue
        wrong = sum(v for k, v in dist.items()
                    if k != CIRCLE and k not in r["wolves"] and k in names)
        if wrong == 0:
            continue
        gi = gp[gid]
        for role, n_seats in Counter(nonwolf).items():
            if role not in rp:
                continue
            obs[rp[role], gi] += sum(v for k, v in dist.items()
                                     if k in names and ends[names.index(k)] == role
                                     and k not in r["wolves"])
            exp[rp[role], gi] += wrong * n_seats / len(nonwolf)
    return obs, exp


all_roles = sorted({role for label in MODELS
                    for _, r in DATA[label]["base_g"].iterrows()
                    for role in r["end_roles"] if role != "Werewolf"})

lift_out = []
for label in MODELS:
    d = DATA[label]
    gids = d["games"]
    ob, eb = lift_matrices(d["base_g"], gids, all_roles)
    of, ef = lift_matrices(d["ft_g"], gids, all_roles)
    rng = np.random.default_rng(LIFT_BOOT_SEED)
    draws = rng.integers(0, len(gids), size=(LIFT_N_BOOT, len(gids)))
    counts = np.stack([np.bincount(row, minlength=len(gids)) for row in draws]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = np.where(counts @ eb.T > 0, (counts @ ob.T) / (counts @ eb.T), np.nan)
        lf = np.where(counts @ ef.T > 0, (counts @ of.T) / (counts @ ef.T), np.nan)
    point_b = np.where(eb.sum(1) > 0, ob.sum(1) / eb.sum(1), np.nan)
    point_f = np.where(ef.sum(1) > 0, of.sum(1) / ef.sum(1), np.nan)
    opp = (eb.sum(1) + ef.sum(1)) / (eb.sum() + ef.sum())
    d_lo = np.nanpercentile(lf - lb, 2.5, axis=0)
    d_hi = np.nanpercentile(lf - lb, 97.5, axis=0)
    for i, role in enumerate(all_roles):
        lift_out.append({"model": label, "role": role, "opportunity_share": opp[i],
                         "base_lift": point_b[i], "ft_lift": point_f[i],
                         "delta": point_f[i] - point_b[i],
                         "d_ci_low": d_lo[i], "d_ci_high": d_hi[i],
                         "sig": bool(d_lo[i] > 0 or d_hi[i] < 0),
                         "in_figure": opp[i] >= MIN_OPPORTUNITY_SHARE})

lift = pd.DataFrame(lift_out)
OUT_DIR = REPO / "analysis" / "cross_model" / "base_vs_ft" / "voting" / PROMPT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
lift.to_csv(OUT_DIR / "lift_base_vs_ft.csv", index=False)
print(f"\nsaved -> {(OUT_DIR / 'lift_base_vs_ft.csv').relative_to(REPO)}")
fig = lift[lift["in_figure"]].copy()
fig["model"] = pd.Categorical(fig["model"], list(MODELS), ordered=True)
fig = fig.sort_values(["role", "model"])
print(f"\nRoles kept (opportunity share >= {MIN_OPPORTUNITY_SHARE:.0%}): {sorted(fig['role'].unique())}")
print("\nSelection lift among incorrect named-player votes (paired bootstrap on the difference):")
print(fig[["role", "model", "opportunity_share", "base_lift", "ft_lift",
           "delta", "d_ci_low", "d_ci_high", "sig"]].round(3).to_string(index=False))
sig = fig[fig["sig"]]
print("\nCells whose BASE->FT change excludes 0:")
print("  (none)" if sig.empty else
      sig[["role", "model", "base_lift", "ft_lift", "delta", "d_ci_low", "d_ci_high"]].round(3).to_string(index=False))

print()
print("=" * 92)
print("D2. START- vs END-ROLE WEREWOLF MISMATCH")
print("=" * 92)


def ww_names(names, roles):
    return {names[i] for i, role in enumerate(roles) if role == "Werewolf"}


srows = []
for label in MODELS:
    d = DATA[label]
    for cond, g in [("BASE", d["base_g"]), ("FT", d["ft_g"])]:
        recs = []
        for gid, r in g.iterrows():
            dist = r["stoch_vote_distribution"]
            if not dist:
                continue
            start_ww = ww_names(r["player_names"], r["start_roles"])
            end_ww = r["wolves"]
            named = {k: v for k, v in dist.items() if k != CIRCLE}
            recs.append({"game_id": gid, "swap_game": bool(start_ww != end_ww),
                         "p_start_not_end": sum(v for k, v in named.items()
                                                if k in start_ww and k not in end_ww),
                         "p_wrong": sum(v for k, v in named.items() if k not in end_ww),
                         "p_correct": r["p_correct"]})
        df = pd.DataFrame(recs)
        srows.append({"model": label, "cond": cond, "n_games": len(df),
                      "n_swap_games": int(df["swap_game"].sum()),
                      "mean_p_start_not_end": df["p_start_not_end"].mean(),
                      "defensible_share_of_wrong": df["p_start_not_end"].sum() / df["p_wrong"].sum(),
                      "acc_swap": df.loc[df["swap_game"], "p_correct"].mean(),
                      "acc_no_swap": df.loc[~df["swap_game"], "p_correct"].mean()})
        DATA[label][f"swap_{cond}"] = df.set_index("game_id").sort_index()

print("\nStart/end Werewolf mismatch, per condition:")
print(pd.DataFrame(srows).round(4).to_string(index=False))

print("\nPaired BASE->FT contrasts:")
rows = []
for label in MODELS:
    b, f = DATA[label]["swap_BASE"], DATA[label]["swap_FT"]
    assert (b.index == f.index).all()
    specs = [("mass on start-only Werewolf", "p_start_not_end", None),
             ("accuracy | swap games", "p_correct", True),
             ("accuracy | no-swap games", "p_correct", False)]
    for metric, col, swap_flag in specs:
        if swap_flag is None:
            vals, n = (f[col] - b[col]).values, len(b)
        else:
            m = b["swap_game"] if swap_flag else ~b["swap_game"]
            vals, n = (f.loc[m, col] - b.loc[m, col]).values, int(m.sum())
        diff, lo, hi = bootstrap_mean_ci(vals)
        rows.append({"model": label, "metric": metric, "n": n, "delta": diff,
                     "ci_low": lo, "ci_high": hi, "sig": bool(lo > 0 or hi < 0)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
