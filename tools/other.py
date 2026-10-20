#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Align and inject GPT-5-mini reports into the 4o-mini file by (subject_id, hadm_id).

Usage:
  python align_gpt5_into_4o.py \
    --base report_subset_10k_gpt4o_mini.csv \
    --src  report_subset_10k_gpt5_mini.csv
"""

import os
import sys
import argparse
import shutil
import pandas as pd

def read_csv_safely(path: str) -> pd.DataFrame:
    # 尝试 UTF-8，不行退回到 latin-1（容错）
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ehr_subset_10k_gpt4o_mini.csv",
                    help="基表：包含 subject_id, hadm_id，需要写入列 report_gpt5_mini")
    ap.add_argument("--src", default="report_subset_10k_gpt5_mini.csv",
                    help="来源表：包含 subject_id, hadm_id, report_gpt5_mini")
    args = ap.parse_args()

    base_path = args.base
    src_path  = args.src

    # 若用户把文件名写成 gpt5o，做一次容错
    if not os.path.exists(src_path):
        alt = "ehr_subset_10k_gpt5_mini.csv"
        if os.path.exists(alt):
            print(f"[info] {src_path} 不存在，改用 {alt}")
            src_path = alt
        else:
            raise FileNotFoundError(f"找不到来源文件：{args.src} 或 {alt}")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"找不到基表文件：{base_path}")

    # 读取
    df_base = read_csv_safely(base_path)
    df_src  = read_csv_safely(src_path)

    # 列检查
    for col in ["subject_id", "hadm_id"]:
        if col not in df_base.columns:
            raise KeyError(f"[base] 缺少列：{col}")
        if col not in df_src.columns:
            raise KeyError(f"[src] 缺少列：{col}")

    if "report_gpt5_mini" not in df_src.columns:
        raise KeyError("[src] 缺少列：report_gpt5_mini")

    # 去重：若来源有重复键，保留最后一条
    df_src = df_src.sort_values(by=["subject_id", "hadm_id"]).drop_duplicates(
        subset=["subject_id", "hadm_id"], keep="last"
    )

    # 只保留需要的列
    df_src_small = df_src[["subject_id", "hadm_id", "report_gpt5_mini"]].copy()

    # 合并（左连接），把 report_gpt5_mini 对齐到基表
    merged = df_base.merge(df_src_small, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_from_gpt5"))


    # 统计对齐数量
    matched = merged["report_gpt5_mini"].notna().sum()
    print(f"[done] 对齐/写入行数：{matched} / {len(merged)}")


    merged.to_csv(base_path, index=False)
    print(f"[write] 已写回：{base_path}")

if __name__ == "__main__":
    main()
