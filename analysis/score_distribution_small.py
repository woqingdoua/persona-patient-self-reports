#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PMF (probability mass function) version:
- 对每个指标(metric)，按评分1..5统计离散概率，画成并排柱状图。
- 仅使用两份文件：
    data/ehr_subset_10k_gpt4o_nopersona.csv
    data/ehr_subset_10k_gpt4o_nopersona_claude.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Config ---------------------------
FILES = [
    "data/ehr_subset_10k_gpt4o_nopersona.csv",        # graded by GPT-4.1
    "data/ehr_subset_10k_gpt4o_nopersona_claude.csv", # graded by Sonnet 3.7
]

METRICS = ["ClinicalCorrectness", "Faithfulness", "Persona", "Realism"]

# 两个打分器两种颜色（与之前一致）
GRADER_COLOR = {
    "GPT-4.1": "#3b5b92",    # 蓝紫
    "Sonnet 3.7": "#d95f02", # 橙色
}

# 输出
OUT_DIR = Path("grade")

# --------------------------- Helpers ---------------------------
def read_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for m in METRICS:
        if m not in df.columns:
            raise KeyError(f"Missing column '{m}' in: {path}")
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df

def parse_meta_from_filename(path: str):
    """返回展示名：grader_disp, label_disp"""
    fname = Path(path).name.lower()
    dataset_disp = "gpt4o-nopersona"  # 仅展示用途
    grader_disp = "Sonnet 3.7" if "claude" in fname else "GPT-4.1"
    label_disp  = f"{grader_disp} grader"
    return grader_disp, label_disp

def pmf_1_to_5(values: np.ndarray) -> np.ndarray:
    """
    统计评分为1..5的概率（直方图密度=False后自行归一化）
    返回 shape=(5,) 的数组，对应分值 1..5 的概率，和为1。
    """
    if values.size == 0:
        return np.zeros(5, dtype=float)
    vals = values[(values >= 1.0) & (values <= 5.0)]
    # 用半开区间边界 0.5,1.5,...,5.5 聚合到 1..5
    bins = np.arange(0.5, 5.6, 1.0)
    counts, _ = np.histogram(vals, bins=bins)
    probs = counts.astype(float) / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    return probs  # 长度=5

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # --------------------------- Load ---------------------------
    datasets = []
    for fp in FILES[:2]:  # 双保险只取两份
        grader_disp, label_disp = parse_meta_from_filename(fp)
        df = read_df(fp)
        datasets.append((grader_disp, label_disp, df))

    # --------------------------- Plot (PMF) ---------------------------
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True, constrained_layout=True)

    x_scores = np.arange(1, 6)        # 1,2,3,4,5
    bar_width = 0.38                  # 两组并排
    offsets = [-bar_width/2, bar_width/2]  # 左右并排偏移

    for ax, metric in zip(axes, METRICS):
        # 为每个数据集计算该指标的 PMF
        pmfs = []
        labels = []
        colors = []
        for grader_disp, label_disp, df in datasets:
            vals = df[metric].dropna().to_numpy(dtype=float)
            pmf = pmf_1_to_5(vals)
            pmfs.append(pmf)
            labels.append(label_disp)
            colors.append(GRADER_COLOR.get(grader_disp, "#333333"))

        # 画并排柱状图
        for i, (pmf, label, color) in enumerate(zip(pmfs, labels, colors)):
            ax.bar(x_scores + offsets[i], pmf, width=bar_width, label=label, alpha=0.95, edgecolor="none")

            # 在柱子顶部加轻微数值标注（可注释掉）
            for xs, p in zip(x_scores + offsets[i], pmf):
                if p > 0:
                    ax.text(xs, p + 0.005, f"{p:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x_scores, [str(s) for s in x_scores])
        ax.set_ylim(0, 1.0)
        ax.set_title(metric, pad=4)
        ax.set_xlabel("Score")
        ax.grid(axis="y", alpha=0.35)
        if ax is axes[0]:
            ax.set_ylabel("Probability")

    # 图例（动态两项）
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    ordered_handles, ordered_labels = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            ordered_handles.append(h)
            ordered_labels.append(lab)

    fig.legend(
        ordered_handles, ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(ordered_labels),
        frameon=False,
    )

    # 保存
    out_png = OUT_DIR / "score_pmf.png"
    out_pdf = OUT_DIR / "score_pmf.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved PMF plots to: {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
