#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

IN_PATH = "result/ehr_subset_10k_gpt.csv"
OUT_DIR = "grade"
OUT_PATH = os.path.join(OUT_DIR, "overall_grade2.csv")

def value_share(series: pd.Series):
    """返回每个取值的 count / share（排除 NaN）和分母 n"""
    s = series.dropna()
    n = len(s)
    vc = s.value_counts()
    shares = (vc / n).rename("share")
    out = pd.concat([vc.rename("count"), shares], axis=1).reset_index(names="label")
    out["denominator"] = n
    return out

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(IN_PATH)
    df = pd.read_csv(IN_PATH)

    for col in ["cmp_winner", "cmp_winner2"]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    # 分别统计两列分布
    dist1 = value_share(df["cmp_winner"])
    dist1.insert(0, "metric", "cmp_winner")

    dist2 = value_share(df["cmp_winner2"])
    dist2.insert(0, "metric", "cmp_winner2")

    # 统计相等/不相等占比（两列都非空时）
    both = df[["cmp_winner", "cmp_winner2"]].dropna()
    denom = len(both)
    eq_count = (both["cmp_winner"] == both["cmp_winner2"]).sum()
    neq_count = denom - eq_count
    agree = pd.DataFrame({
        "metric": ["agreement", "agreement"],
        "label": ["equal", "not_equal"],
        "count": [eq_count, neq_count],
        "share": [eq_count / denom if denom else 0.0,
                  neq_count / denom if denom else 0.0],
        "denominator": [denom, denom],
    })

    # 合并并保存
    out = pd.concat([dist1, dist2, agree], ignore_index=True)
    out["share"] = out["share"].round(4)  # 可按需调整精度

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[done] Saved to {OUT_PATH}")
    print(out)

if __name__ == "__main__":
    main()
