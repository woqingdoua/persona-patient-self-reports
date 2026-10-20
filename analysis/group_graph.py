#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Config --------------------
CSV = "result/ehr_subset_10k_gpt.csv"
OUT_DIR = "grade"

# persona 字段（从 persona JSON 中解析）
PERSONA_KEYS = [
    "drg_severity", "insurance", "admission_location", "admission_type",
    "gender", "anchor_age", "language", "marital_status", "education_level"
]

# “长尾”折叠阈值（最多显示的 top K；其余合并为 Other）
TOPK_LIMITS = {
    "language": 10,
    "admission_type": 6,
    "admission_location": 6,
}

# -------------------- Utils --------------------
def read_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "persona" not in df.columns:
        raise RuntimeError("CSV 里需要有 `persona` 列（JSON 字符串）。")
    # 统一 winner 列名
    if "cmp_winner" not in df.columns or "cmp_winner2" not in df.columns:
        raise RuntimeError("CSV 里需要 `cmp_winner` 和 `cmp_winner2` 两列。")
    return df

def parse_persona_json(s):
    if pd.isna(s):
        return {}
    try:
        # 兼容双引号嵌套等轻微格式问题
        return json.loads(s)
    except Exception:
        # 尝试修复常见转义
        ss = re.sub(r"''", '"', str(s)).replace("None", "null")
        try:
            return json.loads(ss)
        except Exception:
            return {}

def normalize_winner(w):
    if pd.isna(w):
        return "tie"
    s = str(w).strip().lower().replace("-", "_")
    if "5" in s:
        return "5m"
    if "4" in s or "4o" in s:
        return "4o"
    if "tie" in s or s in {"=", "equal"}:
        return "tie"
    # 兜底：未知视为 tie
    return "tie"

def bin_age(val):
    try:
        a = float(val)
    except Exception:
        return "NA"
    if a <= 30:  return "≤30"
    if a <= 45:  return "31–45"
    if a <= 60:  return "46–60"
    if a <= 75:  return "61–75"
    return "≥76"

def fold_long_tail(s: pd.Series, key: str) -> pd.Series:
    """按频次保留 Top-K，其余为 'Other'."""
    k = TOPK_LIMITS.get(key, None)
    if not k or s.nunique(dropna=False) <= k:
        return s.fillna("NA")
    top = s.value_counts(dropna=False).index[:k].tolist()
    return s.apply(lambda x: x if x in top else "Other").fillna("NA")

def prepare_persona_columns(df: pd.DataFrame) -> pd.DataFrame:
    pj = df["persona"].apply(parse_persona_json)
    for k in PERSONA_KEYS:
        df[k] = pj.apply(lambda d: d.get(k, np.nan))
    # 特殊处理
    df["anchor_age"] = df["anchor_age"].apply(bin_age)
    for col in ["language", "admission_type", "admission_location"]:
        df[col] = fold_long_tail(df[col].astype(str), col)
    # 统一字符串
    for k in PERSONA_KEYS:
        df[k] = df[k].astype(str).replace({"nan": "NA", "None": "NA"})
    return df

def winrate_5m(series: pd.Series) -> float:
    """去掉 tie 后 5m 的获胜率。"""
    s = series.dropna().map(normalize_winner)
    non_tie = s[s != "tie"]
    if len(non_tie) == 0:
        return np.nan
    return float((non_tie == "5m").mean())

def non_tie_n(series: pd.Series) -> int:
    s = series.dropna().map(normalize_winner)
    return int((s != "tie").sum())

# -------------------- Core computation --------------------
def compute_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    返回宽表：每个 persona 属性 × 取值一行，
    包含两个评审的 winrate_5m（去 tie）、non_tie 样本量，以及差值 |Δ|。
    """
    out = []
    for attr in PERSONA_KEYS:
        grp = df.groupby(attr, dropna=False)
        for val, sub in grp:
            r4 = winrate_5m(sub["cmp_winner"])
            rC = winrate_5m(sub["cmp_winner2"])
            n4 = non_tie_n(sub["cmp_winner"])
            nC = non_tie_n(sub["cmp_winner2"])
            n_all = len(sub)
            out.append({
                "attribute": attr,
                "value": str(val),
                "n": int(n_all),
                "n_non_tie_gpt41": n4,
                "n_non_tie_claude": nC,
                "winrate_5m_gpt41": r4,
                "winrate_5m_claude": rC,
                "diff_signed": None if (pd.isna(r4) or pd.isna(rC)) else (rC - r4),
                "diff_abs": None if (pd.isna(r4) or pd.isna(rC)) else abs(rC - r4),
            })
    res = pd.DataFrame(out)
    # 排序：先按属性，再按 |Δ| 从大到小
    res["diff_abs"] = pd.to_numeric(res["diff_abs"], errors="coerce")
    res.sort_values(by=["attribute", "diff_abs"], ascending=[True, False], inplace=True)
    return res

# -------------------- Plotting --------------------
def plot_disagreement_bars(df_stat: pd.DataFrame, out_png: str, out_pdf: str):
    """
    每个 persona 属性 1 个子图：横轴为取值，纵轴为 |Δ|（两评审的 GPT-5-mini 获胜率差异，去 tie）。
    """
    attrs = PERSONA_KEYS
    n_attr = len(attrs)
    n_cols = 3
    n_rows = int(np.ceil(n_attr / n_cols))

    plt.rcParams.update({
        "figure.figsize": (18, 10),
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharey=True)
    ymax = 1.0

    for i, attr in enumerate(attrs):
        ax = axes[i // n_cols, i % n_cols]
        sub = df_stat[df_stat["attribute"] == attr].copy()
        # 为了可读性，可限制最多显示前 12 个差异最大的取值
        sub = sub.head(12)
        x = np.arange(len(sub))
        ax.bar(x, sub["diff_abs"].fillna(0.0), color="#3b5b92", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["value"], rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(attr.replace("_", " "))
        # 在柱状上方标注样本量 n（全部样本）
        for xi, (v, n) in enumerate(zip(sub["diff_abs"], sub["n"])):
            if pd.isna(v):
                continue
            ax.text(xi, min(v + 0.04, ymax), f"{v:.2f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8)

    # 清空多余子图
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle("Disagreement between graders on GPT-5-mini win rate (ties removed)\n"
                 "|Δ| = | winrate_5m(Claude 3.7) − winrate_5m(GPT-4.1) |", y=1.02, fontsize=14)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_ylabel("|Δ| (win rate difference)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# -------------------- Main --------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = read_df(CSV)
    df = prepare_persona_columns(df)

    # 计算分歧
    stat = compute_disagreement(df)

    # 导出统计表
    csv_out = os.path.join(OUT_DIR, "persona_disagreement_stats.csv")
    stat.to_csv(csv_out, index=False)
    print(f"[saved] {csv_out}  （包含各 persona 取值的样本量、两评审 winrate 与差值）")

    # 出图：每个 persona 一个子图，展示 |Δ|
    png = os.path.join(OUT_DIR, "persona_disagreement_bars.png")
    pdf = os.path.join(OUT_DIR, "persona_disagreement_bars.pdf")
    plot_disagreement_bars(stat, png, pdf)
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")

if __name__ == "__main__":
    main()
