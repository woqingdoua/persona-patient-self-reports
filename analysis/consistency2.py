#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

FILE_4O_GPT    = "result/ehr_subset_10k_gpt4o_mini.csv"
FILE_4O_CLAUDE = "result/ehr_subset_10k_gpt4o_mini_claude.csv"
FILE_5M_GPT    = "result/ehr_subset_10k_gpt5_mini.csv"
FILE_5M_CLAUDE = "result/ehr_subset_10k_gpt5_mini_claude.csv"

METRICS = ["ClinicalCorrectness", "Faithfulness", "Persona", "Realism"]
COLOR = {"4o-mini": "#3b5b92", "5-mini": "#d95f02"}
OUT_DIR = "grade"

try:
    from scipy import stats
    from scipy.integrate import cumtrapz
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_GAUSS = True
except Exception:
    HAS_GAUSS = False

def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def align_pair(df_gpt, df_cld):
    keys = ["subject_id", "hadm_id"]
    for k in keys:
        if k not in df_gpt.columns or k not in df_cld.columns:
            raise KeyError(f"Missing merge key: {k}")
    if df_gpt.duplicated(keys).any():
        df_gpt = df_gpt.drop_duplicates(keys, keep="first")
    if df_cld.duplicated(keys).any():
        df_cld = df_cld.drop_duplicates(keys, keep="first")
    return pd.merge(df_gpt, df_cld, on=keys, how="inner", suffixes=("_gpt", "_cld"))

def get_cols(df, metric):
    cand_gpt = [f"{metric}_gpt", f"gpt_{metric}", f"{metric}_gpt41"]
    cand_cld = [f"{metric}_cld", f"cld_{metric}", f"{metric}_claude"]
    col_gpt = next((c for c in cand_gpt if c in df.columns), None)
    col_cld = next((c for c in cand_cld if c in df.columns), None)
    if col_gpt is None or col_cld is None:
        raise KeyError(f"Cannot find columns for {metric} (gpt/cld).")
    x = pd.to_numeric(df[col_gpt], errors="coerce")
    y = pd.to_numeric(df[col_cld], errors="coerce")
    m = x.notna() & y.notna()
    return x[m].to_numpy(), y[m].to_numpy()

def smooth_cdf(abs_diff: np.ndarray, xs: np.ndarray) -> np.ndarray:
    if abs_diff.size == 0:
        return np.zeros_like(xs)
    if HAS_SCIPY:
        if np.std(abs_diff) == 0:
            abs_diff = abs_diff + np.random.normal(0, 1e-6, size=abs_diff.shape)
        kde = stats.gaussian_kde(abs_diff, bw_method="scott")
        dens = kde.evaluate(xs)
        cdf = cumtrapz(dens, xs, initial=0)
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return np.clip(cdf, 0, 1)
    # fallback：直方图+平滑+积分
    bins = max(100, int(np.sqrt(abs_diff.size)) * 4)
    hist, bin_edges = np.histogram(abs_diff, bins=bins, range=(0, xs.max()), density=True)
    if HAS_GAUSS:
        hist = gaussian_filter1d(hist, sigma=2, mode="nearest")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dens_interp = np.interp(xs, centers, hist, left=0, right=0)
    cdf = np.cumsum(dens_interp) * (xs[1] - xs[0])
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return np.clip(cdf, 0, 1)

def first_x_at_y(xs: np.ndarray, ys: np.ndarray, y_target: float) -> float:
    idx = np.argmax(ys >= y_target)
    return float(xs[idx]) if ys[idx] >= y_target else float("nan")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_4o = align_pair(read_csv(FILE_4O_GPT), read_csv(FILE_4O_CLAUDE))
    df_5m = align_pair(read_csv(FILE_5M_GPT), read_csv(FILE_5M_CLAUDE))
    print(f"Aligned rows: GPT-4o-mini={len(df_4o)}, GPT-5-mini={len(df_5m)}")

    # 全局 x 上限
    diffs_all = []
    for metric in METRICS:
        g1, c1 = get_cols(df_4o, metric)
        g2, c2 = get_cols(df_5m, metric)
        diffs_all.extend([np.abs(g1 - c1), np.abs(g2 - c2)])
    max_diff = float(np.nanmax(np.concatenate(diffs_all))) if diffs_all else 2.0
    x_max = max(1.0, min(3.0, np.ceil(max_diff * 10) / 10.0))
    xs = np.linspace(0.0, x_max, 600)

    # 1×4 子图
    plt.rcParams.update({"figure.figsize": (18, 4.8), "axes.grid": True})
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)

    target_y = 0.90
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        g4, c4 = get_cols(df_4o, metric)
        g5, c5 = get_cols(df_5m, metric)
        a4, a5 = np.abs(g4 - c4), np.abs(g5 - c5)

        y4 = smooth_cdf(a4, xs)
        y5 = smooth_cdf(a5, xs)

        ax.plot(xs, y4, lw=2, color=COLOR["4o-mini"], label="GPT-4o-mini")
        ax.plot(xs, y5, lw=2, color=COLOR["5-mini"], label="GPT-5-mini")

        # y=0.90 水平线 & x90 竖线（不标注数值）
        ax.axhline(target_y, ls="--", lw=1, color="gray")
        x90_4 = first_x_at_y(xs, y4, target_y)
        x90_5 = first_x_at_y(xs, y5, target_y)
        if np.isfinite(x90_4):
            ax.axvline(x90_4, ls=":", lw=1.2, color=COLOR["4o-mini"])
        if np.isfinite(x90_5):
            ax.axvline(x90_5, ls=":", lw=1.2, color=COLOR["5-mini"])

        ax.set_title(metric, fontsize=16)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if i == 0:
            ax.set_ylabel("Proportion  P(|diff| ≤ x)", fontsize=14)
        # 每个子图都显示 X 轴标题
        ax.set_xlabel("Threshold x", fontsize=14)

    # 图例移到整张图正下方
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               ncol=2, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=14)

    # 为底部图例预留空间
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    png = os.path.join(OUT_DIR, "agreement_curves.png")
    pdf = os.path.join(OUT_DIR, "agreement_curves.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved → {png} and {pdf}")

if __name__ == "__main__":
    main()
