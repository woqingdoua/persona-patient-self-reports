#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paired tests & robustness checks:
Compare GPT-5-mini vs GPT-4o-mini reports under two graders (GPT-4.1, Claude 3.7).
Outputs CSV tables with significance tests, effect sizes, agreement metrics,
plus robustness (outlier removal) and optional persona stratification.
"""

import os
import numpy as np
import pandas as pd

from scipy import stats

# -------------------- Inputs --------------------
FILE_4O_GPT    = "result/ehr_subset_10k_gpt4o_mini.csv"          # GPT-4.1 → scores on 4o-mini reports
FILE_4O_CLAUDE = "result/ehr_subset_10k_gpt4o_mini_claude.csv"   # Claude 3.7 → scores on 4o-mini reports
FILE_5M_GPT    = "result/ehr_subset_10k_gpt5_mini.csv"           # GPT-4.1 → scores on 5-mini reports
FILE_5M_CLAUDE = "result/ehr_subset_10k_gpt5_mini_claude.csv"    # Claude 3.7 → scores on 5-mini reports

METRICS = ["ClinicalCorrectness", "Faithfulness", "Persona", "Realism"]
AGREE_TAU = 0.50
OUT_DIR = "grade"

# -------------------- Utils --------------------
def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def align_pair(df_4o, df_5m, keys=("subject_id","hadm_id")):
    # keep unique keys
    for df in (df_4o, df_5m):
        if pd.Series(df[k] for k in keys).isna().any():
            pass
        if df.duplicated(list(keys)).any():
            df.drop_duplicates(list(keys), keep="first", inplace=True)
    df = pd.merge(df_4o, df_5m, on=list(keys), how="inner", suffixes=("_4o", "_5m"))
    return df

def get_cols(df, metric, grader_tag):  # grader_tag in {"gpt","claude"}
    # Try common variants
    cand_4o = [f"{metric}_{grader_tag}_4o", f"{metric}_4o_{grader_tag}",
               f"{metric}_4o", f"{grader_tag}_{metric}_4o", f"{grader_tag}_{metric}"]
    cand_5m = [f"{metric}_{grader_tag}_5m", f"{metric}_5m_{grader_tag}",
               f"{metric}_5m", f"{grader_tag}_{metric}_5m"]

    col_4o = next((c for c in cand_4o if c in df.columns), None)
    col_5m = next((c for c in cand_5m if c in df.columns), None)
    if col_4o is None or col_5m is None:
        raise KeyError(f"Cannot find columns for metric={metric}, grader={grader_tag}")
    x = pd.to_numeric(df[col_5m], errors="coerce")
    y = pd.to_numeric(df[col_4o], errors="coerce")
    m = x.notna() & y.notna()
    return (x[m] - y[m]).to_numpy()  # Δ = 5m - 4o

def cohens_d_paired(d):  # d_i = paired differences
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 2: return np.nan
    return float(d.mean() / d.std(ddof=1))

def cliffs_delta_paired(d):
    # δ = (#positive - #negative)/n_non_tie  (ties excluded)
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0: return np.nan
    pos = np.sum(d > 0)
    neg = np.sum(d < 0)
    n_eff = pos + neg
    return float((pos - neg) / n_eff) if n_eff > 0 else 0.0

def ba_metrics(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(bias=np.nan, loa_low=np.nan, loa_high=np.nan)
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if d.size>1 else 0.0
    loa_low, loa_high = bias - 1.96*sd, bias + 1.96*sd
    return dict(bias=bias, loa_low=loa_low, loa_high=loa_high)

def core_stats(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {}
    # tests
    t_stat, t_p = stats.ttest_rel(d, np.zeros_like(d))  # equivalent to one-sample t on diff
    # Wilcoxon (ignores zeros); use 'pratt' to handle zeros more gracefully (SciPy ≥1.11)
    try:
        w_stat, w_p = stats.wilcoxon(d, zero_method='pratt', alternative='two-sided', method='approx')
    except TypeError:
        w_stat, w_p = stats.wilcoxon(d)  # fallback
    # effect sizes
    cohen_d = cohens_d_paired(d)
    cliffs = cliffs_delta_paired(d)
    # agreement-style metrics
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    agree = float((np.abs(d) <= AGREE_TAU).mean())
    # inverse CDF thresholds
    x80 = float(np.quantile(np.abs(d), 0.80))
    x90 = float(np.quantile(np.abs(d), 0.90))
    # Bland–Altman
    ba = ba_metrics(d)
    # 95% CI for mean diff (t CI)
    n = d.size
    se = np.std(d, ddof=1)/np.sqrt(n) if n>1 else np.nan
    tcrit = stats.t.ppf(0.975, df=n-1) if n>1 else np.nan
    mean_ci_low = float(d.mean() - tcrit*se) if n>1 else np.nan
    mean_ci_high= float(d.mean() + tcrit*se) if n>1 else np.nan
    return dict(
        n=int(n),
        mean_diff=float(d.mean()),
        mean_diff_ci_low=mean_ci_low,
        mean_diff_ci_high=mean_ci_high,
        t_stat=t_stat, t_p=t_p,
        wilcoxon=W_stat_to_float(w_stat), wilcoxon_p=w_p,
        cohen_d=cohen_d, cliffs_delta=cliffs,
        mae=mae, rmse=rmse, agree_prop_leq_tau=agree, x80=x80, x90=x90,
        **ba
    )

def W_stat_to_float(x):
    try: return float(x)
    except Exception: return np.nan

def summarize_block(df_pair, grader_tag, metric):
    d = get_cols(df_pair, metric, grader_tag)  # Δ array
    base = core_stats(d)
    base.update(dict(grader=("GPT-4.1" if grader_tag=="gpt" else "Claude 3.7"),
                     metric=metric, tau=AGREE_TAU, subset="all"))
    # robustness 1: remove outliers |Δ|>2
    mask = np.abs(d) <= 2.0
    r1 = core_stats(d[mask])
    r1.update(dict(grader=("GPT-4.1" if grader_tag=="gpt" else "Claude 3.7"),
                   metric=metric, tau=AGREE_TAU, subset="no_outlier(|Δ|≤2)"))
    return base, r1, d

# -------------------- Main --------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read & align within each grader, then merge 4o vs 5m by keys
    df_4o_gpt = read_csv(FILE_4O_GPT)
    df_5m_gpt = read_csv(FILE_5M_GPT)
    df_4o_cld = read_csv(FILE_4O_CLAUDE)
    df_5m_cld = read_csv(FILE_5M_CLAUDE)

    # Align per grader (to preserve possible grader-specific columns)
    pair_gpt = align_pair(df_4o_gpt, df_5m_gpt)       # columns ..._4o and ..._5m
    pair_cld = align_pair(df_4o_cld, df_5m_cld)

    rows_main, rows_robust, rows_persona = [], [], []

    # Loop metrics & graders
    for metric in METRICS:
        # GPT-4.1 grader
        base, r1, d_gpt = summarize_block(pair_gpt, "gpt", metric)
        rows_main.append(base); rows_robust.append(r1)

        # Claude 3.7 grader
        base, r1, d_cld = summarize_block(pair_cld, "claude", metric)
        rows_main.append(base); rows_robust.append(r1)

        # Persona stratification if possible
        for tag, pair_df, d_vec, grader_name in [
            ("GPT-4.1", pair_gpt, d_gpt, "GPT-4.1"),
            ("Claude 3.7", pair_cld, d_cld, "Claude 3.7"),
        ]:
            if "persona" in pair_df.columns:
                for p, sub in pair_df.groupby("persona"):
                    try:
                        d_sub = get_cols(sub, metric, "gpt" if tag=="GPT-4.1" else "claude")
                    except KeyError:
                        continue
                    stat = core_stats(d_sub)
                    stat.update(dict(grader=grader_name, metric=metric, subset=f"persona={p}", tau=AGREE_TAU))
                    rows_persona.append(stat)

    # Save CSVs
    df_main = pd.DataFrame(rows_main)
    df_robust = pd.DataFrame(rows_robust)
    df_persona = pd.DataFrame(rows_persona)

    df_main.to_csv(os.path.join(OUT_DIR, "paired_tests_main.csv"), index=False)
    df_robust.to_csv(os.path.join(OUT_DIR, "paired_tests_robust_outlier_removed.csv"), index=False)
    if len(df_persona):
        df_persona.to_csv(os.path.join(OUT_DIR, "paired_tests_by_persona.csv"), index=False)

    # Console summary
    print("=== MAIN (Δ = 5-mini − 4o-mini) ===")
    for g in ["GPT-4.1","Claude 3.7"]:
        sub = df_main[df_main["grader"]==g]
        print(f"\n-- {g} --")
        for m in METRICS:
            row = sub[sub["metric"]==m].iloc[0]
            print(f"{m:20s}  n={row['n']:5.0f}  meanΔ={row['mean_diff']:+.3f} "
                  f"[{row['mean_diff_ci_low']:+.3f},{row['mean_diff_ci_high']:+.3f}]  "
                  f"Wilcoxon p={row['wilcoxon_p']:.3g}  d={row['cohen_d']:.2f}  "
                  f"agree@{AGREE_TAU}={row['agree_prop_leq_tau']:.3f}  x90={row['x90']:.2f}")

    print("\nSaved CSVs to:", OUT_DIR)

if __name__ == "__main__":
    main()
