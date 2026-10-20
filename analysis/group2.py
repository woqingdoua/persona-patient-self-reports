#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ===== 输入文件 =====
FILES = {
    "gpt4o_gpt41":  "data/ehr_subset_10k_gpt4o_mini.csv",
    "gpt4o_claude": "data/ehr_subset_10k_gpt4o_mini_claude.csv",
    "gpt5_gpt41":   "data/ehr_subset_10k_gpt5_mini.csv",
    "gpt5_claude":  "data/ehr_subset_10k_gpt5_mini_claude.csv",
}

# ===== 配置 =====
MERGE_KEYS = ["subject_id", "hadm_id"]
ATTRS = ["drg_severity","insurance","admission_location","admission_type",
         "gender","anchor_age","language","marital_status","education_level"]
METRICS = ["Persona", "Realism"]
TOPK = {"language": 10, "admission_type": 6, "admission_location": 6}
OUT_DIR = "grade"
COLORS = {"gpt4o":"#3b5b92","gpt5":"#d95f02"}

# ---------- 基础工具 ----------
def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_persona_json_cell(cell):
    """把 `"{""gender"": ""M""...}"` 这种字符串安全解析为 dict。"""
    if not isinstance(cell, str) or not cell.strip():
        return {}
    s = cell.strip()
    # 修正双双引号
    if '""' in s and s.count('""') >= 2:
        s = s.replace('""','"')
    try:
        return json.loads(s)
    except Exception:
        # 再尝试去掉外层引号
        try:
            if s[0] in ("'", '"') and s[-1] == s[0]:
                return json.loads(s[1:-1])
        except Exception:
            return {}

def ensure_persona_columns(df: pd.DataFrame) -> pd.DataFrame:
    """若缺少 ATTRS，就尝试从 persona/persona_json 展开。"""
    have = all(a in df.columns for a in ATTRS)
    if have:
        return df

    src_col = None
    for cand in ["persona", "persona_json", "persona_str"]:
        if cand in df.columns:
            src_col = cand; break
    if src_col is None:
        return df  # 没法展开，后续会提示

    # 逐行解析
    recs = df[src_col].apply(parse_persona_json_cell)
    # 生成各属性列
    for a in ATTRS:
        df[a] = recs.apply(lambda d: d.get(a, np.nan))
    return df

def bin_anchor_age(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    bins  = [-np.inf, 30, 45, 60, 75, np.inf]
    labels= ["≤30","31–45","46–60","61–75","≥76"]
    return pd.cut(s, bins=bins, labels=labels, right=True, include_lowest=True)

def cap_topk(series: pd.Series, k: int) -> pd.Series:
    counts = series.value_counts(dropna=False)
    keep = set(counts.head(k).index.tolist())
    return series.where(series.isin(keep), other="Other")

def normalize_persona_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "anchor_age" in out: out["anchor_age"] = bin_anchor_age(out["anchor_age"])
    for col, k in TOPK.items():
        if col in out: out[col] = cap_topk(out[col].astype(str), k)
    for a in ATTRS:
        if a in out and a not in TOPK:
            out[a] = out[a].astype(str)
    return out

def align_reports(df4o, df5, metric: str):
    # 确保 persona 列存在（从 JSON 展开）
    df4o = ensure_persona_columns(df4o)
    df5  = ensure_persona_columns(df5)
    # 主键对齐
    for k in MERGE_KEYS:
        if k not in df4o or k not in df5:
            raise KeyError(f"Missing merge key {k} in CSVs.")
    df4o = df4o.drop_duplicates(MERGE_KEYS, keep="first")
    df5  = df5.drop_duplicates(MERGE_KEYS, keep="first")

    need4 = MERGE_KEYS + ATTRS + [metric]
    need5 = MERGE_KEYS + ATTRS + [metric]
    df4o_ = df4o[[c for c in need4 if c in df4o.columns]].copy()
    df5_  = df5 [[c for c in need5 if c in df5.columns]].copy()
    df4o_.rename(columns={metric: f"{metric}_gpt4o"}, inplace=True)
    df5_.rename(columns={metric: f"{metric}_gpt5"},  inplace=True)
    df = pd.merge(df4o_, df5_, on=MERGE_KEYS, how="inner", suffixes=("","_dup"))
    # 删除 5 的重复 persona 列
    df.drop(columns=[c for c in df.columns if c.endswith("_dup")], inplace=True, errors="ignore")
    return df

def bar_with_ci(ax, categories, mean_4o, mean_5m, sem_4o, sem_5m, title=""):
    x = np.arange(len(categories)); w = 0.38
    ax.bar(x-w/2, mean_4o, width=w, color=COLORS["gpt4o"], label="GPT-4o-mini", yerr=sem_4o, capsize=2)
    ax.bar(x+w/2, mean_5m, width=w, color=COLORS["gpt5"],  label="GPT-5-mini",  yerr=sem_5m, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories], rotation=25, ha="right")
    ax.set_ylim(0, 5.05)
    ax.set_ylabel("Mean score (1–5)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", ls=":", alpha=0.3)

def plot_for_grader(grader_name, metric, df4o, df5, out_dir):
    df = align_reports(df4o, df5, metric)
    # 若还是缺 persona，给出友好提示
    if not all(a in df.columns for a in ATTRS):
        missing = [a for a in ATTRS if a not in df.columns]
        raise RuntimeError(f"Persona columns missing even after expand: {missing}. "
                           f"Check persona JSON column exists and is parseable.")

    df = normalize_persona_columns(df)

    plt.rcParams.update({"figure.figsize": (18, 10)})
    fig, axes = plt.subplots(3, 3, sharey=True)
    axes = axes.ravel()

    for i, a in enumerate(ATTRS):
        ax = axes[i]
        grp = df.groupby(a, dropna=False)
        counts = grp.size().sort_values(ascending=False)
        cats = [c for c, n in counts.items() if n >= 20]  # 最少 20 个样本
        if len(cats) == 0:
            ax.set_axis_off(); ax.text(0.5,0.5,f"No data: {a}",ha="center",va="center"); continue

        agg = grp[[f"{metric}_gpt4o", f"{metric}_gpt5"]].agg(["mean","count","std"])
        agg = agg.loc[cats]

        mean_4o = agg[(f"{metric}_gpt4o","mean")].astype(float)
        mean_5m = agg[(f"{metric}_gpt5","mean")].astype(float)
        sem_4o  = (agg[(f"{metric}_gpt4o","std")] / np.sqrt(agg[(f"{metric}_gpt4o","count")].clip(lower=1))).astype(float)
        sem_5m  = (agg[(f"{metric}_gpt5","std")] / np.sqrt(agg[(f"{metric}_gpt5","count")].clip(lower=1))).astype(float)

        bar_with_ci(ax, cats, mean_4o.loc[cats], mean_5m.loc[cats], sem_4o.loc[cats], sem_5m.loc[cats],
                    title=a.replace("_"," "))

    fig.suptitle(f"{grader_name} — {metric}: report-wise comparison by persona groups", fontsize=16, y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.97),fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.98])

    os.makedirs(out_dir, exist_ok=True)
    tag = grader_name.replace(" ","_").lower() + "_" + metric.lower()
    fig.savefig(os.path.join(out_dir, f"{tag}_by_persona.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{tag}_by_persona.pdf"), bbox_inches="tight")
    print(f"[saved] {os.path.join(out_dir, f'{tag}_by_persona.png')}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    g4o_g41 = read_csv(FILES["gpt4o_gpt41"])
    g4o_cld = read_csv(FILES["gpt4o_claude"])
    g5_g41  = read_csv(FILES["gpt5_gpt41"])
    g5_cld  = read_csv(FILES["gpt5_claude"])

    # GPT-4.1 作为打分器
    for metric in METRICS:
        plot_for_grader("GPT-4.1", metric, g4o_g41, g5_g41, OUT_DIR)

    # Claude 3.7 作为打分器
    for metric in METRICS:
        plot_for_grader("Claude 3.7", metric, g4o_cld, g5_cld, OUT_DIR)

if __name__ == "__main__":
    main()
