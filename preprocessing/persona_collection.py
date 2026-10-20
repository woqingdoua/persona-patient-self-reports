#MIMIC-IV 3.1

#admissions.csv.gz:  marital_status, admission_location, admission_type
#hosp/patients.csv.gz： gender, anchor_age
#hosp/drgcodes.csv.gz: drg_severity (Severity of Illness)

'''
admission_type:
emergency：URGENT, EW EMER., DIRECT EMER.
observation：include any OBSERVATION、EU OBSERVATION、DIRECT OBSERVATION、AMBULATORY OBSERVATION
elective：ELECTIVE, SURGICAL SAME DAY ADMISSION
'''


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build persona.csv by joining:
- hosp/admissions.csv.gz: subject_id, hadm_id, marital_status, admission_location, admission_type
- hosp/drgcodes.csv.gz:   subject_id, hadm_id, drg_severity (prefer APR-DRG; take max per hadm_id)
- patient_with_edu.csv:   subject_id, gender, anchor_age, education_level

Output columns (CSV):
subject_id, hadm_id, marital_status, insurance, admission_location, admission_type,
drg_severity, gender, anchor_age, education_level
"""

import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Join admissions + drgcodes + patient_with_edu to persona.csv")
    ap.add_argument("--admissions", default="mimic/admissions.csv.gz", help="Path to hosp/admissions.csv.gz")
    ap.add_argument("--drgcodes",   default="mimic/drgcodes.csv.gz",   help="Path to hosp/drgcodes.csv.gz")
    ap.add_argument("--patients_edu", default="patient_with_edu.csv", help="Path to patient_with_edu.csv")
    ap.add_argument("--output",     default="persona.csv",       help="Output CSV path")
    args = ap.parse_args()

    # --- 1) admissions: only needed columns ---
    adm_usecols = ["subject_id", "hadm_id", "marital_status", "insurance","language", "admission_location", "admission_type"]
    adm = pd.read_csv(args.admissions, usecols=adm_usecols)

    # 保证唯一 (subject_id, hadm_id)（有些站点不会重复，这里做个保险）
    adm = adm.drop_duplicates(subset=["subject_id", "hadm_id"])

    # --- 2) drgcodes: prefer APR-DRG, max severity per (subject_id, hadm_id) ---
    # 一些版本的列名大小写一致；若不存在，会抛异常提醒
    drg_usecols = ["subject_id", "hadm_id", "drg_type", "drg_severity"]
    drg = pd.read_csv(args.drgcodes, usecols=drg_usecols)

    # 只在 APR-DRG 下 drg_severity 才有意义
    drg_apr = drg.loc[drg["drg_type"].str.upper() == "APR"].copy()

    # 转成数值（可能是字符串），无效值设为 NaN
    drg_apr["drg_severity"] = pd.to_numeric(drg_apr["drg_severity"], errors="coerce")

    # 同一次住院可能有多条 APR-DRG，取最大严重度代表该次住院
    drg_agg = (
        drg_apr.groupby(["subject_id", "hadm_id"], as_index=False)["drg_severity"]
        .max()
    )

    # --- 3) merge admissions + drg (left join: 以 admissions 为基准) ---
    df = adm.merge(drg_agg, on=["subject_id", "hadm_id"], how="left")

    # --- 4) patients_with_edu: subject-level attributes ---
    edu_usecols = ["subject_id", "gender", "anchor_age", "education_level"]
    edu = pd.read_csv(args.patients_edu, usecols=edu_usecols)

    # 若 patient_with_edu 有重复 subject_id，保留第一次出现
    edu = edu.drop_duplicates(subset=["subject_id"])

    # 合并（按 subject_id）
    df = df.merge(edu, on="subject_id", how="left")

    # --- 5) 选列并导出 ---
    cols = [
        "subject_id", "hadm_id",
        "marital_status","insurance","language", "admission_location", "admission_type",
        "drg_severity",
        "gender", "anchor_age", "education_level",
    ]
    # 确保列存在（万一上游文件某列缺失会抛 KeyError，便于尽早发现问题）
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after merge: {missing}")

    out = df[cols].copy()

    # 可选：统一类型
    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").astype("Int64")
    out["hadm_id"]    = pd.to_numeric(out["hadm_id"], errors="coerce").astype("Int64")
    # drg_severity 是 1-4，保留可空整数
    out["drg_severity"] = pd.to_numeric(out["drg_severity"], errors="coerce").astype("Int64")

    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}  shape={out.shape}")
    # Patient-level dataframe preview omitted in public release.

if __name__ == "__main__":
    main()

