#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

FILES = [
    "result/ehr_subset_10k_gpt4o_nopersona.csv",
    "result/ehr_subset_10k_gpt4o_nopersona_claude.csv",
    "result/ehr_subset_10k_gpt4o_mini.csv",
    "result/ehr_subset_10k_gpt4o_mini_claude.csv",
    "result/ehr_subset_10k_gpt5_mini.csv",
    "result/ehr_subset_10k_gpt5_mini_claude.csv",
]

OUT_PATH = "grade/overall_grade1.csv"
EXCLUDE = {"subject_id", "hadm_id"}  # 一定排除

def summarize_file(path: str) -> pd.Series:
    """返回：该文件每个指标的 'mean ± var' 字符串（index=指标名）。"""
    df = pd.read_csv(path)

    # 仅数值列，并排除 subject_id、hadm_id
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in EXCLUDE]
    if not num_cols:
        return pd.Series(dtype="object", name=os.path.basename(path))

    means = df[num_cols].mean(numeric_only=True)
    vars_ = df[num_cols].var(ddof=1, numeric_only=True)  # 样本方差

    # 组装为 mean ± var（保留两位小数）
    formatted = []
    for col in num_cols:
        m = float(means[col]) if pd.notna(means[col]) else float("nan")
        v = float(vars_[col]) if pd.notna(vars_[col]) else float("nan")
        formatted.append(f"{m:.2f} ± {v:.2f}")
    s = pd.Series(formatted, index=num_cols, name=os.path.basename(path))
    return s

def main():
    rows = []
    all_cols = set()

    for f in FILES:
        if not os.path.exists(f):
            print(f"[warn] not found, skip: {f}")
            continue
        s = summarize_file(f)
        rows.append(s)
        all_cols.update(s.index.tolist())

    if not rows:
        print("[error] no files summarized.")
        return

    # 统一列集合并对齐（缺失填空字符串）
    all_cols = sorted(all_cols)
    table = pd.DataFrame(index=[s.name for s in rows], columns=all_cols, dtype="object")
    for s in rows:
        table.loc[s.name, s.index] = s.values
    table = table.fillna("")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    table.to_csv(OUT_PATH, index=True)
    print(f"[done] saved: {OUT_PATH}")
    # Patient-level dataframe preview omitted in public release.

if __name__ == "__main__":
    main()
