#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ========== 配置 ==========
FILE_4O_GPT    = "result/ehr_subset_10k_gpt4o_mini.csv"                # GPT-4.1 → 4o-mini
FILE_4O_CLAUDE = "result/ehr_subset_10k_gpt4o_mini_claude.csv"         # Claude 3.7 → 4o-mini
FILE_5M_GPT    = "result/ehr_subset_10k_gpt5_mini.csv"                 # GPT-4.1 → 5-mini
FILE_5M_CLAUDE = "result/ehr_subset_10k_gpt5_mini_claude.csv"          # Claude 3.7 → 5-mini
# 新增：4o-mini（去 persona 版本）
FILE_4O_NOP_GPT    = "result/ehr_subset_10k_gpt4o_nopersona.csv"            # GPT-4.1 → 4o-nopersona
FILE_4O_NOP_CLAUDE = "result/ehr_subset_10k_gpt4o_nopersona_claude.csv"     # Claude 3.7 → 4o-nopersona

MERGE_KEYS = ["subject_id", "hadm_id"]
METRICS = ["ClinicalCorrectness", "Faithfulness", "Persona", "Realism", "overall"]  # 如需包含 overall 可加上 "overall"

# Bland–Altman：差异定义为 GPT-4.1 − Claude（你也可改为 Claude − GPT-4.1）
DIFF_SIGN = +1    # +1: diff = gpt41 - claude；-1: diff = claude - gpt41
THRESH = 0.5      # “一致率”阈值：|diff| ≤ THRESH
KAPPA_BIN = 0.25  # 计算加权 κ 时的量化步长（将连续分数量化后计算）

OUT_DIR = "grade"
PNG_PATH = os.path.join(OUT_DIR, "consistency.png")
PDF_PATH = os.path.join(OUT_DIR, "consistency.pdf")
CSV_PATH = os.path.join(OUT_DIR, "consistency_metrics.csv")

# 依赖（可选）
try:
    import pingouin as pg
    HAS_PG = True
except Exception:
    HAS_PG = False

try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SK = True
except Exception:
    HAS_SK = False


# ========== 工具 ==========
def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def align_pair(df_gpt, df_cld):
    """按 subject_id, hadm_id 一一对齐；保留公共行。"""
    for k in MERGE_KEYS:
        if k not in df_gpt.columns or k not in df_cld.columns:
            raise KeyError(f"缺少键列 {MERGE_KEYS}，请检查输入文件。")
    # 可能有重复键：去重保留首个
    if df_gpt.duplicated(MERGE_KEYS).any():
        df_gpt = df_gpt.drop_duplicates(MERGE_KEYS, keep="first")
    if df_cld.duplicated(MERGE_KEYS).any():
        df_cld = df_cld.drop_duplicates(MERGE_KEYS, keep="first")
    df = pd.merge(df_gpt, df_cld, on=MERGE_KEYS, how="inner", suffixes=("_gpt41", "_claude"))
    return df

def to_float(s):
    return pd.to_numeric(s, errors="coerce")

def lin_ccc(x, y):
    """Lin's Concordance Correlation Coefficient."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return np.nan
    return (2 * sxy) / denom

def icc_2_1(x, y):
    """
    ICC(2,1): 双因素随机效应，单次测量。
    如果安装了 pingouin，调用它；否则用简易实现（两列评分）。
    """
    if HAS_PG:
        df_long = pd.DataFrame({"targets": np.arange(len(x)), "rater": "A", "score": x})
        df_long = pd.concat([df_long,
                             pd.DataFrame({"targets": np.arange(len(y)), "rater": "B", "score": y})], ignore_index=True)
        icc_tbl = pg.intraclass_corr(data=df_long, targets='targets', raters='rater', ratings='score')
        row = icc_tbl.loc[icc_tbl["Type"] == "ICC2", :]
        return float(row["ICC"].values[0])
    # 简易两评审实现（近似）
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    k = 2
    n = len(x)
    m_i = np.vstack([x, y]).T.mean(axis=1)
    m_dot = np.mean(np.hstack([x, y]))
    SS_between = k * np.sum((m_i - m_dot) ** 2)
    SS_resid = np.sum((x - m_i) ** 2 + (y - m_i) ** 2)
    MS_between = SS_between / (n - 1)
    MS_resid = SS_resid / (n * (k - 1))
    return (MS_between - MS_resid) / (MS_between + (k - 1) * MS_resid)

def weighted_kappa_continuous(x, y, bin_size=0.25):
    """将连续分数量化后，计算二次加权 κ（需要 sklearn）。"""
    if not HAS_SK:
        return np.nan
    xq = np.round(to_float(x) / bin_size).astype("Int64")
    yq = np.round(to_float(y) / bin_size).astype("Int64")
    mask = xq.notna() & yq.notna()
    if mask.sum() < 2:
        return np.nan
    return float(cohen_kappa_score(xq[mask].astype(int), yq[mask].astype(int), weights="quadratic"))

def compute_metrics(x, y, thresh=THRESH, bin_size=KAPPA_BIN):
    """返回一组一致性指标。"""
    x = to_float(x); y = to_float(y)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return {}

    diff = DIFF_SIGN * (x - y)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    agree_prop = float((np.abs(diff) <= thresh).mean())

    # 相关性
    pear = float(np.corrcoef(x, y)[0, 1]) if x.std(ddof=0) > 0 and y.std(ddof=0) > 0 else np.nan
    try:
        from scipy.stats import spearmanr, kendalltau
        spear = float(spearmanr(x, y, nan_policy="omit").correlation)
        kend = float(kendalltau(x, y, variant="b").correlation)
    except Exception:
        spear, kend = np.nan, np.nan

    ccc = lin_ccc(x, y)
    icc = icc_2_1(x, y)
    kappa_q = weighted_kappa_continuous(x, y, bin_size=bin_size)

    # Bland–Altman 参数
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    loa_low = mean_diff - 1.96 * sd_diff if np.isfinite(sd_diff) else np.nan
    loa_high = mean_diff + 1.96 * sd_diff if np.isfinite(sd_diff) else np.nan

    return {
        "n": int(len(x)),
        "pearson_r": pear,
        "spearman_r": spear,
        "kendall_tau_b": kend,
        "ccc": ccc,
        "icc_2_1": icc,
        "weighted_kappa_q": kappa_q,
        "mae": mae,
        "rmse": rmse,
        "bias_mean_diff": mean_diff,
        "loa_low": loa_low,
        "loa_high": loa_high,
        "agree_prop_|diff|<=%.2f" % THRESH: agree_prop,
    }

def bland_altman_ax(ax, x, y, title):
    """绘制单面板 Bland–Altman：x=均值，y=差值（gpt41−claude）。"""
    x = to_float(x); y = to_float(y)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    mean_xy = (x + y) / 2.0
    diff = DIFF_SIGN * (x - y)  # GPT-4.1 − Claude
    m = float(np.mean(diff))
    s = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    low, high = m - 1.96 * s, m + 1.96 * s

    ax.scatter(mean_xy, diff, s=9, alpha=0.35, edgecolors="none")
    ax.axhline(m, color="tab:orange", linestyle="-", linewidth=1.2, label=f"mean={m:.2f}")
    if np.isfinite(s) and s > 0:
        ax.axhline(low, color="gray", linestyle="--", linewidth=1.0, label=f"LoA−={low:.2f}")
        ax.axhline(high, color="gray", linestyle="--", linewidth=1.0, label=f"LoA+={high:.2f}")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Mean of two graders")
    ax.set_ylabel("GPT-4.1 − Claude 3.7")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.text(0.98, 0.05, f"n={len(diff)}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))


# ========== 主流程 ==========
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读取与对齐
    df_4o_gpt = read_csv(FILE_4O_GPT)
    df_4o_cld = read_csv(FILE_4O_CLAUDE)
    df_5m_gpt = read_csv(FILE_5M_GPT)
    df_5m_cld = read_csv(FILE_5M_CLAUDE)
    df_4onop_gpt = read_csv(FILE_4O_NOP_GPT)
    df_4onop_cld = read_csv(FILE_4O_NOP_CLAUDE)

    df_4o = align_pair(df_4o_gpt, df_4o_cld)
    df_5m = align_pair(df_5m_gpt, df_5m_cld)
    df_4onop = align_pair(df_4onop_gpt, df_4onop_cld)
    print(f"Aligned rows: 4o-mini={len(df_4o)}, 4o-nopersona={len(df_4onop)}, 5-mini={len(df_5m)}")

    # 需要绘制/统计的三组报告（行顺序）
    report_specs = [
        ("gpt4o-mini",       df_4o,     "GPT-4o-mini"),
        ("gpt4o-nopersona",  df_4onop,  "GPT-4o-mini (no persona)"),
        ("gpt5-mini",        df_5m,     "GPT-5-mini"),
    ]

    # 画图：3 行 × len(METRICS)
    plt.rcParams.update({
        "figure.figsize": (16, 11),  # 稍高，容纳 3 行
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(len(report_specs), len(METRICS), sharex=False, sharey=False)
    fig.suptitle("Bland–Altman: GPT-4.1 vs Claude 3.7 (diff = GPT-4.1 − Claude)",
                 fontsize=14, y=0.98)

    metrics_rows = []

    # 逐行逐列绘制与统计
    for i, (rep_key, df_rep, rep_title) in enumerate(report_specs):
        for j, m in enumerate(METRICS):
            # 兼容两种可能的列命名
            if f"{m}_gpt41" in df_rep.columns and f"{m}_claude" in df_rep.columns:
                x = df_rep[f"{m}_gpt41"]
                y = df_rep[f"{m}_claude"]
            else:
                x = df_rep.get(f"gpt_{m}", df_rep.get(f"{m}_gpt"))   # 兜底
                y = df_rep.get(f"cld_{m}", df_rep.get(f"{m}_cld"))   # 兜底

            bland_altman_ax(axes[i, j], x, y, f"{rep_title} • {m}")

            stats = compute_metrics(x, y)
            stats.update({"report": rep_key, "metric": m})
            metrics_rows.append(stats)

    # 图例仅放第一行第一个子图
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Saved plots → {PNG_PATH}, {PDF_PATH}")

    # 指标导出
    df_metrics = pd.DataFrame(metrics_rows)
    cols = ["report", "metric", "n",
            "pearson_r", "spearman_r", "kendall_tau_b",
            "ccc", "icc_2_1", "weighted_kappa_q",
            "mae", "rmse",
            "bias_mean_diff", "loa_low", "loa_high",
            f"agree_prop_|diff|<={THRESH:.2f}"]
    df_metrics = df_metrics[[c for c in cols if c in df_metrics.columns]]
    df_metrics.to_csv(CSV_PATH, index=False)
    print(f"Saved metrics → {CSV_PATH}")

    # 额外提示
    if not HAS_PG:
        print("Note: 未安装 'pingouin'，ICC(2,1) 为近似或 N/A。建议 pip install pingouin")
    if not HAS_SK:
        print("Note: 未安装 'scikit-learn'，weighted kappa 为 N/A。建议 pip install scikit-learn")

if __name__ == "__main__":
    main()
