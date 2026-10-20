#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ───────── 配置 ─────────
FILES = {
    "gpt4o_gpt41": "result/ehr_subset_10k_gpt4o_mini.csv",
    "gpt4o_claude": "result/ehr_subset_10k_gpt4o_mini_claude.csv",
    "gpt5_gpt41": "result/ehr_subset_10k_gpt5_mini.csv",
    "gpt5_claude": "result/ehr_subset_10k_gpt5_mini_claude.csv",
}

COLOR_4O   = "#3b5b92"  # 负差：4o-mini 更高
COLOR_5MINI = "#d95f02" # 正差：5-mini 更高

MERGE_KEYS = ["subject_id", "hadm_id"]
METRICS = ["ClinicalCorrectness", "Faithfulness", "Persona", "Realism"]
BIN_WIDTH = 0.25     # 分箱宽度（根据你的评分粒度调整：0.25 或 0.5 都可）
ALPHA = 0.85
OUT_DIR = "grade"
PNG_PATH = os.path.join(OUT_DIR, "diff_dists_row2x4.png")
PDF_PATH = os.path.join(OUT_DIR, "diff_dists_row2x4.pdf")

# ───────── 工具函数 ─────────
def read_csv_safe(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_keys(df: pd.DataFrame, name: str):
    missing = [k for k in MERGE_KEYS if k not in df.columns]
    if missing:
        raise KeyError(f"{name} 缺少匹配键列: {missing}，需要 {MERGE_KEYS}")

def to_numeric_cols(df: pd.DataFrame, cols: list):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def align_reports_for_grader(df_4o: pd.DataFrame, df_5m: pd.DataFrame, grader_name: str) -> pd.DataFrame:
    """
    对同一位评审者，将其对 4o-mini 与 5-mini 报告的评分在键列上对齐。
    返回列：MERGE_KEYS + 每个 metric 的 diff 列（<metric>__diff = 5-mini - 4o-mini）
    """
    ensure_keys(df_4o, f"{grader_name} × 4o-mini")
    ensure_keys(df_5m, f"{grader_name} × 5-mini")

    cols_needed = MERGE_KEYS + METRICS
    a = df_4o[[c for c in cols_needed if c in df_4o.columns]].copy()
    b = df_5m[[c for c in cols_needed if c in df_5m.columns]].copy()

    to_numeric_cols(a, METRICS)
    to_numeric_cols(b, METRICS)

    if a.duplicated(MERGE_KEYS).any():
        a = a.drop_duplicates(MERGE_KEYS, keep="first")
    if b.duplicated(MERGE_KEYS).any():
        b = b.drop_duplicates(MERGE_KEYS, keep="first")

    merged = pd.merge(a, b, on=MERGE_KEYS, how="inner", suffixes=("_4o", "_5m"), validate="one_to_one")

    out = merged[MERGE_KEYS].copy()
    for m in METRICS:
        out[f"{m}__diff"] = merged[f"{m}_5m"] - merged[f"{m}_4o"]
    return out

def nice_bins_from_data(diffs: np.ndarray, bin_width: float):
    """根据数据确定紧凑的 bins：按 bin_width 向外取整，避免空白区间。"""
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.array([-1, 0, 1])  # 兜底
    dmin, dmax = diffs.min(), diffs.max()
    xmin = np.floor(dmin / bin_width) * bin_width
    xmax = np.ceil(dmax / bin_width) * bin_width
    # 至少留 1 个 bin 的外边距
    pad = bin_width * 0.0
    xmin -= pad
    xmax += pad
    # 生成边界（包含右端）
    edges = np.arange(xmin, xmax + bin_width * 1.0001, bin_width)
    # 防止意外只有一个边
    if len(edges) < 3:
        edges = np.array([xmin - bin_width, xmin, xmin + bin_width])
    return edges

def percent(y, _pos):
    return f"{100*y:.0f}%"

def plot_diff_hist(ax, diffs: np.ndarray, title: str):
    """
    画“比例直方图”：每个 bin 的高度为该 bin 样本比例（sum=1），
    负区间着色 COLOR_4O、正区间着色 COLOR_5MINI；竖线 x=0；X 轴自适应。
    """
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    bins = nice_bins_from_data(diffs, BIN_WIDTH)

    # 用 weights 把柱高转为“比例”而不是 density
    weights = np.ones_like(diffs) / len(diffs)
    neg_mask = diffs < 0
    pos_mask = ~neg_mask  # 包含 0

    # 先算整体的最大比例，用来设置 y 轴上限
    total_prop_per_bin, _ = np.histogram(diffs, bins=bins, weights=weights)
    ymax = total_prop_per_bin.max() * 1.15 if total_prop_per_bin.size else 1.0

    ax.hist(diffs[neg_mask], bins=bins, weights=weights[neg_mask],
            color=COLOR_4O, alpha=ALPHA, label="4o-mini higher (<0)")
    ax.hist(diffs[pos_mask], bins=bins, weights=weights[pos_mask],
            color=COLOR_5MINI, alpha=ALPHA, label="5-mini higher (≥0)")

    # 竖直虚线 x=0
    ax.axvline(0.0, linestyle="--", linewidth=1, color="black")

    # 坐标
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, max(0.01, ymax))
    ax.yaxis.set_major_formatter(FuncFormatter(percent))
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Score difference: (5-mini) − (4o-mini)")

    # 标注均值
    mean_diff = float(np.nanmean(diffs))
    ax.set_title(f"{title}\nmean Δ = {mean_diff:.3f}")

    # 细网格提高可读性
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读取四个文件
    g4o_gpt41 = read_csv_safe(FILES["gpt4o_gpt41"])
    g4o_claude = read_csv_safe(FILES["gpt4o_claude"])
    g5m_gpt41 = read_csv_safe(FILES["gpt5_gpt41"])
    g5m_claude = read_csv_safe(FILES["gpt5_claude"])

    # 对齐为两位评审者的“5-mini − 4o-mini”分差
    df_diff_gpt41 = align_reports_for_grader(g4o_gpt41, g5m_gpt41, "GPT-4.1")
    df_diff_claude = align_reports_for_grader(g4o_claude, g5m_claude, "Claude 3.7")

    # 作图：2 行（评审者） × 4 列（指标）
    fig, axes = plt.subplots(2, len(METRICS), figsize=(18, 9), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    # 第 1 行：GPT-4.1 评审
    for j, m in enumerate(METRICS):
        diffs = df_diff_gpt41[f"{m}__diff"].to_numpy(dtype=float)
        plot_diff_hist(axes[0, j], diffs, f"GPT-4.1 grader — {m}")

    # 第 2 行：Claude 3.7 评审
    for j, m in enumerate(METRICS):
        diffs = df_diff_claude[f"{m}__diff"].to_numpy(dtype=float)
        plot_diff_hist(axes[1, j], diffs, f"Claude 3.7 grader — {m}")

    # 统一图例（放到左上）
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, loc="upper left")

    fig.suptitle("Distribution of Score Differences (5-mini − 4o-mini)\nRight area larger ⇒ 5-mini systematically higher", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(PNG_PATH)
    fig.savefig(PDF_PATH)
    print(f"Saved to:\n - {PNG_PATH}\n - {PDF_PATH}")

if __name__ == "__main__":
    main()
