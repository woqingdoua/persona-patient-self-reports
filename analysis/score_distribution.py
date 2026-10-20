#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot score distributions (KDE) for 4 metrics in a 1x4 figure.
- Same report uses the same color; different graders use different linestyles.
- Files expected:
    result/ehr_subset_10k_gpt4o_mini.csv
    result/ehr_subset_10k_gpt4o_mini_claude.csv
    result/ehr_subset_10k_gpt5_mini.csv
    result/ehr_subset_10k_gpt5_mini_claude.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# --------------------------- Config ---------------------------
FILES = [
    "result/ehr_subset_10k_gpt4o_mini.csv",
    "result/ehr_subset_10k_gpt4o_mini_claude.csv",
    "result/ehr_subset_10k_gpt5_mini.csv",
    "result/ehr_subset_10k_gpt5_mini_claude.csv",
]

METRICS = ["ClinicalCorrectness", "Faithfulness", "Persona", "Realism"]

# 颜色按“报告”分配；线型按“打分模型(Grader)”分配
REPORT_COLOR = {
    "4o-mini": "#3b5b92",   # 蓝紫色
    "5-mini": "#d95f02",    # 暖橙
}
GRADER_STYLE = {"GPT-4.1": "-", "Sonnet 3.7": "--"}         # 注意：展示名用 GPT-4.1

# KDE 网格 & 平滑
X = np.linspace(1.0, 5.0, 600)    # 评分范围 1–5
BANDWIDTH = 0.20                  # 可按需要调小(更细)或调大(更平滑)

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
    """
    返回:
      report_key: '4o-mini' / '5-mini'
      grader_disp: 'GPT-4.1' / 'Sonnet 3.7'   (仅用于展示)
      label_disp:  'GPT-4o-mini (GPT-4.1)' 等
    """
    fname = Path(path).name.lower()
    if "gpt4o_mini" in fname:
        report_key, report_disp = "4o-mini", "GPT-4o-mini"
    elif "gpt5_mini" in fname:
        report_key, report_disp = "5-mini", "GPT-5-mini"
    else:
        report_key, report_disp = ("4o-mini", "GPT-4o-mini") if "4o" in fname else ("5-mini", "GPT-5-mini")

    grader_disp = "Sonnet 3.7" if "claude" in fname else "GPT-4.1"  # 展示用：按你的命名
    label_disp  = f"{report_disp} ({grader_disp})"
    return report_key, grader_disp, label_disp

def kde_gaussian(values: np.ndarray, x: np.ndarray, bw: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(x)
    diffs = (x[:, None] - values[None, :]) / bw
    kernel = np.exp(-0.5 * diffs * diffs) / (np.sqrt(2 * np.pi) * bw)
    dens = kernel.mean(axis=1)
    area = np.trapz(dens, x)
    return dens / (area if area > 0 else 1.0)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # --------------------------- Load ---------------------------
    datasets = []
    for fp in FILES:
        report_key, grader_disp, label_disp = parse_meta_from_filename(fp)
        df = read_df(fp)
        datasets.append((report_key, grader_disp, label_disp, df))

    # --------------------------- Plot ---------------------------
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.6,
    })

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=False, constrained_layout=True)

    for ax, metric in zip(axes, METRICS):
        for report_key, grader_disp, label_disp, df in datasets:
            vals = df[metric].dropna().to_numpy(dtype=float)
            vals = vals[(vals >= 1.0) & (vals <= 5.0)]
            kde = kde_gaussian(vals, X, BANDWIDTH)
            ax.plot(
                X, kde,
                color=REPORT_COLOR.get(report_key, "#333333"),
                linestyle=GRADER_STYLE.get(grader_disp, "-"),
                label=label_disp,  # 用于后面收集句柄
            )

            # —— 已按要求删除均值竖线 ——（原 ax.axvline(...) 已移除）

        ax.set_title(metric, pad=4)
        ax.set_xlabel("Score (1–5)")
        ax.set_xlim(1, 5)
        ax.grid(True, alpha=0.35)
        if ax is axes[0]:
            ax.set_ylabel("Density")

    # -------- 单个总图例（按 desired_order 排序） --------
    desired_order = [
        "GPT-4o-mini (GPT-4.1)",
        "GPT-4o-mini (Sonnet 3.7)",
        "GPT-5-mini (GPT-4.1)",
        "GPT-5-mini (Sonnet 3.7)",
    ]

    # 从第一幅子图收集所有句柄与标签
    handles, labels = axes[0].get_legend_handles_labels()
    lab2handle = {lab: h for h, lab in zip(handles, labels)}
    ordered_handles = [lab2handle[l] for l in desired_order if l in lab2handle]
    ordered_labels  = [l for l in desired_order if l in lab2handle]

    # 放在图下方（不改变布局的其它部分）
    fig.legend(
        ordered_handles, ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.0),
        ncol=4,
        frameon=False,
    )

    #fig.suptitle("Score distributions by report (color) and grader (linestyle)", fontsize=15, y=1.05)




    fig.savefig("grade/score_dists.png", bbox_inches="tight")
    fig.savefig("grade/score_dists.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

