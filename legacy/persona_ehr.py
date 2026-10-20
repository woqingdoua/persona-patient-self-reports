#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json, re
import pandas as pd
import numpy as np

PERSONA_KEYS = [
    "marital_status", "admission_location", "admission_type",
    "drg_severity", "gender", "anchor_age", "education_level"
]

# ====== 新增：分割标记 ======
HEAD_SPLIT_MARKERS = [
    "Physical Exam",
    "Pertinent Results",
    "Pertinent Labs",
    "Pertinent Micro",
    "Pertinent Imaging",
]
# 允许大小写不敏感、并容忍标记后面的 ":" / "-" / 换行等
MARKER_PATTERNS = [
    re.compile(rf"\b{re.escape(m)}\b\s*[:\-\n]?", flags=re.IGNORECASE)
    for m in HEAD_SPLIT_MARKERS
]

def coerce_ids(df, subj_col="subject_id", hadm_col="hadm_id"):
    """把ID列转为可空整数，避免字符串/浮点导致的合并失败。"""
    for c in [subj_col, hadm_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def safe_read_csv(path, usecols=None):
    """优先按 usecols 读取；若列名不完全匹配，则读取全量后再子集。"""
    try:
        return pd.read_csv(path, usecols=usecols)
    except Exception:
        df = pd.read_csv(path)
        if usecols is not None:
            cols = [c for c in usecols if c in df.columns]
            if len(cols) < (len(usecols) if usecols else 0):
                missing = set(usecols) - set(cols)
                print(f"[warn] {path} 缺少列：{missing}，将忽略缺失列")
            df = df[cols] if cols else df
        return df

def _strip_leading_newline(x):
    """去掉文本开头的换行符（\n / \r\n / \r）；非字符串保持原值。"""
    if isinstance(x, str):
        x = x.replace('\r\n', '\n').replace('\r', '\n').lstrip('\n')
    return x

# ====== 新增：按多个标记截断并取更短的上半段 ======
def _upper_half_shortest_by_markers(text: str) -> str:
    """
    给定原始出院小结文本：
      - 对每个标记（Physical Exam / Pertinent *），找到第一次出现位置；
      - 取该位置之前的“上半段”作为一个候选；
      - 在所有候选里选择**更短**的那一个；若一个标记都没匹配，则返回原文。
    """
    if not isinstance(text, str) or not text:
        return text
    s = text.replace('\r\n', '\n').replace('\r', '\n')
    candidates = []
    for pat in MARKER_PATTERNS:
        m = pat.search(s)
        if m:
            candidates.append(s[:m.start()])
    if candidates:
        # “拿个更短取哪个” → 选择长度最小的上半段
        chosen = min(candidates, key=lambda t: len(t) if t is not None else float('inf'))
        return chosen.strip()
    return s.strip()

def load_discharge(discharge_path):
    """
    读取出院小结文本。优先用 'text' 列；如果叫别名（如 'note_text'），自动兜底。
    若同一次住院有多条，取文本最长的一条；并清理首行换行符。
    之后按照多个标记（Physical Exam / Pertinent *）截断并取更短上半段作为最终 text。
    """
    df = safe_read_csv(discharge_path)
    # 识别文本列名
    text_col = None
    for cand in ["text", "note_text", "notes", "discharge_text"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise KeyError(f"{discharge_path} 未找到文本列（期望 'text' 或类似列名）")

    need = ["subject_id", "hadm_id", text_col]
    df = df[[c for c in need if c in df.columns]].copy()
    if "subject_id" not in df.columns or "hadm_id" not in df.columns:
        raise KeyError(f"{discharge_path} 需要包含 subject_id 与 hadm_id")

    # 先去掉开头换行
    df[text_col] = df[text_col].apply(_strip_leading_newline)

    df = coerce_ids(df)
    # 取每个 (subject_id, hadm_id) 文本最长的一条（避免按字节切错记录）
    df["_len"] = df[text_col].astype(str).str.len()
    df = df.sort_values("_len", ascending=False).drop_duplicates(
        subset=["subject_id", "hadm_id"], keep="first"
    ).drop(columns=["_len"])

    # === 关键：对选中的文本做“多标记截断并取更短上半段” ===
    df[text_col] = df[text_col].apply(_upper_half_shortest_by_markers)

    df = df.rename(columns={text_col: "text"})
    return df

def build_persona_row(row):
    """把 persona 相关列按要求打包为字典，去掉 NaN。"""
    d = {}
    for k in PERSONA_KEYS:
        if k in row and pd.notna(row[k]):
            val = row[k]
            if pd.api.types.is_integer_dtype(type(val)) or isinstance(val, (np.integer,)):
                val = int(val)
            d[k] = val
    return d

def main():
    ap = argparse.ArgumentParser(description="Build persona_self_report.csv by joining reports + persona + discharge text")
    ap.add_argument("--train_csv", default="data/diagnoses.csv",
                    help="path to mimic_diagnosis_icd11(test).csv")
    ap.add_argument("--persona_csv", default="persona.csv", help="path to persona.csv")
    ap.add_argument("--discharge", default="mimic/discharge.csv.gzv",
                    help="path to MIMIC-IV-Note discharge file (default tries .gzv, falls back to .gz)")
    ap.add_argument("--output", default="persona_self_report.csv", help="output CSV path")
    args = ap.parse_args()

    # 1) 读取报告
    rep_cols = ["subject_id", "hadm_id", "report", "icd11_title"]
    reports = safe_read_csv(args.train_csv, usecols=rep_cols)
    reports = coerce_ids(reports)
    missing = [c for c in rep_cols if c not in reports.columns]
    if missing:
        raise KeyError(f"{args.train_csv} 缺少列：{missing}")

    # 2) 读取 persona
    per_cols = ["subject_id", "hadm_id"] + PERSONA_KEYS
    persona = safe_read_csv(args.persona_csv, usecols=per_cols)
    persona = coerce_ids(persona).drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")

    # 合并报告与 persona（左连接）
    df = reports.merge(persona, on=["subject_id", "hadm_id"], how="left")

    # persona 序列化
    df["persona"] = df.apply(build_persona_row, axis=1).apply(lambda d: json.dumps(d, ensure_ascii=False))

    # 3) 读取 discharge 文本（优先 .gzv，不存在则尝试 .gz）
    discharge_path = args.discharge
    if not os.path.exists(discharge_path) and discharge_path.endswith(".gzv"):
        alt = discharge_path[:-1]
        if os.path.exists(alt):
            print(f"[info] {discharge_path} 不存在，改用 {alt}")
            discharge_path = alt

    discharge = load_discharge(discharge_path)[["subject_id", "hadm_id", "text"]].copy()

    # 4) 合并文本
    df = df.merge(discharge, on=["subject_id", "hadm_id"], how="left")

    # 5) 输出
    out_cols = ["subject_id", "hadm_id", "icd11_title", "report", "persona", "text"]
    out = coerce_ids(df[out_cols].copy())
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}  shape={out.shape}")
    print(len(out))
    # Patient-level dataframe preview omitted in public release.

if __name__ == "__main__":
    main()
