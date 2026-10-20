#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, numpy as np, pandas as pd

# 六个教育层级（可按需改名）
EDU_CATS = ["primary", "middle", "high", "college", "masters", "phd"]

# 按年龄段的条件先验（示例占位，请按项目需要调整）
# 面向 25 岁及以上美国总体（BASE_PRIOR）
# 顺序: [小学, 初中, 高中, 大学(不含硕博), 硕士, 博士(含专业博士)]
BASE_PRIOR = [4.7, 5.2, 27.5, 47.6, 12.0, 3.0]  # ≈100

# 分年龄的 AGE_PRIOR（按 anchor_age 归档）
AGE_PRIOR = {
    "(0,12]":   [95.0, 5.0, 0.0, 0.0, 0.0, 0.0],   # 小学为主；见 NCES 对美小学/初中/高中年级段定义
    "(12,16]":  [10.0, 80.0, 10.0, 0.0, 0.0, 0.0], # 初中为主，部分进入高中
    "(16,18]":  [2.0, 15.0, 80.0, 3.0, 0.0, 0.0],  # 高中为主，少量提前修大学学分
    "(18,24]":  [1.0, 6.0, 38.0, 54.4, 0.5, 0.1],  # 多数在读/仅高中；少数已获学士；极少硕博
    "(25,64]":  [4.5, 5.0, 27.5, 49.0, 11.5, 2.5], # 主力劳动年龄，学位占比更高
    "(65,200]": [5.0, 6.0, 34.0, 44.0, 8.5, 2.5],  # 老年组 BA+较低，<HS 略高
}

def _normalize(p):
    p = np.array(p, dtype=float)
    p = np.clip(p, 0, None)
    return (p / p.sum()) if p.sum() > 0 else np.ones_like(p)/len(p)

def choose_age_bin(age):
    bins = [(0,12),(12,16),(16,18),(18,23),(23,65),(65,200)]
    labels = ["(0,12]","(12,16]","(16,18]","(18,23]","(23,65]","(65,200]"]
    for (lo, hi), lab in zip(bins, labels):
        if age > lo and age <= hi:
            return lab
    return "(23,65]"

def dirichlet_perturb(p, alpha, rng):
    """给先验一点随机性；alpha越大扰动越小（建议200~500）。"""
    p = _normalize(p)
    return rng.dirichlet(alpha * p)

def sample_one(age, rng, add_noise=True, alpha=300, guard=True):
    age = float(age) if pd.notna(age) else 40.0
    p = AGE_PRIOR.get(choose_age_bin(age), BASE_PRIOR)

    # 未成年人护栏：不采到大学及以上
    if guard and age <= 18:
        mask = np.array([1,1,1,0,0,0])
        p = _normalize(np.array(p) * mask)

    if add_noise:
        p = dirichlet_perturb(p, alpha=alpha, rng=rng)
    else:
        p = _normalize(p)

    return rng.choice(EDU_CATS, p=p)

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic education level from anchor_age.")
    ap.add_argument("--input",  default="patients.csv.gz", help="path to patients.csv.gz")
    ap.add_argument("--output", default="patient_with_edu.csv", help="output CSV path")
    ap.add_argument("--seed",   type=int, default=2025, help="random seed for reproducibility")
    ap.add_argument("--alpha",  type=float, default=300.0, help="Dirichlet noise strength (higher=more stable)")
    ap.add_argument("--no-noise", action="store_true", help="disable Dirichlet perturbation")
    args = ap.parse_args()

    # 读取（pandas 会自动识别 .gz）
    usecols = ["subject_id", "gender", "anchor_age"]
    df = pd.read_csv(args.input, usecols=usecols)

    rng = np.random.default_rng(args.seed)
    df["education"] = [
        sample_one(age=a, rng=rng, add_noise=(not args.no_noise), alpha=args.alpha, guard=True)
        for a in df["anchor_age"].tolist()
    ]

    out = df.rename(columns={"education": "education_level"})
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}\nRows: {len(out)}")
    # Patient-level dataframe preview omitted in public release.

if __name__ == "__main__":
    main()
