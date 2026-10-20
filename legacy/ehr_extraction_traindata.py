#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json, re, random
import pandas as pd
import numpy as np

PERSONA_KEYS = [
    "marital_status", "admission_location", "admission_type",
    "drg_severity", "gender", "anchor_age", "education_level"
]

# ====== 分割标记 ======
HEAD_SPLIT_MARKERS = [
    "Physical Exam",
    "Pertinent Results",
    "Pertinent Labs",
    "Pertinent Micro",
    "Pertinent Imaging",
]
MARKER_PATTERNS = [
    re.compile(rf"\b{re.escape(m)}\b\s*[:\-\n]?", flags=re.IGNORECASE)
    for m in HEAD_SPLIT_MARKERS
]

# ====== Chief Complaint ======
CC_PATTERN = re.compile(r'(?is)chief\s*complaints?\s*[:：]\s*')

MAX_TOKENS = 700  # 只保留前 700 个 token
DEFAULT_SEED = 2025

# -------------------- I/O & Utils --------------------
def coerce_ids(df, subj_col="subject_id", hadm_col="hadm_id"):
    for c in [subj_col, hadm_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def safe_read_csv(path, usecols=None):
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
    if isinstance(x, str):
        x = x.replace('\r\n', '\n').replace('\r', '\n').lstrip('\n')
    return x

def _upper_half_shortest_by_markers(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    s = text.replace('\r\n', '\n').replace('\r', '\n')
    candidates = []
    for pat in MARKER_PATTERNS:
        m = pat.search(s)
        if m:
            candidates.append(s[:m.start()])
    if candidates:
        chosen = min(candidates, key=lambda t: len(t) if t is not None else float('inf'))
        return chosen.strip()
    return s.strip()

def _cut_before_chief_complaint(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    s = text
    m = CC_PATTERN.search(s)
    if not m:
        return s.strip()
    after = s[m.end():]
    return after.strip()

def _take_first_tokens(text: str, n: int = MAX_TOKENS) -> str:
    if not isinstance(text, str) or not text:
        return text
    toks = re.findall(r"\S+", text)
    if len(toks) <= n:
        return " ".join(toks).strip()
    return " ".join(toks[:n]).strip()

def load_discharge(discharge_path):
    df = safe_read_csv(discharge_path)
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

    df[text_col] = df[text_col].apply(_strip_leading_newline)

    df = coerce_ids(df)
    df["_len"] = df[text_col].astype(str).str.len()
    df = df.sort_values("_len", ascending=False).drop_duplicates(
        subset=["subject_id", "hadm_id"], keep="first"
    ).drop(columns=["_len"])

    df[text_col] = (df[text_col]
                    .apply(_upper_half_shortest_by_markers)
                    .apply(_cut_before_chief_complaint)
                    .apply(lambda x: _take_first_tokens(x, MAX_TOKENS)))
    # 全部转小写
    df[text_col] = df[text_col].astype(str).str.lower()

    df = df.rename(columns={text_col: "text"})
    return df

# -------------------- Persona --------------------
def build_persona_row(row):
    d = {}
    for k in PERSONA_KEYS:
        if k in row and pd.notna(row[k]):
            val = row[k]
            # 小写字符串值
            if isinstance(val, str):
                val = val.lower()
            elif pd.api.types.is_integer_dtype(type(val)) or isinstance(val, (np.integer,)):
                val = int(val)
            d[k] = val
    return d

# -------------------- Balanced Sampling --------------------
def stratified_balanced_sample(df, label_col, total_n=20000, seed=DEFAULT_SEED):
    """
    尽量让每个类别样本数接近的分层均衡抽样。
    - 若类别数 > total_n：随机选 total_n 个类别，每类取 1。
    - 否则：每类先取 base = total_n // n_classes；
      类内不足则全取；聚合后若总数 < total_n，再按类剩余容量随机补齐到 total_n。
    """
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=[label_col]).copy()
    classes = df[label_col].unique().tolist()
    n_classes = len(classes)

    if n_classes == 0:
        raise ValueError("没有可用的类别（icd11_title 全部为空？）")

    # 极端：类别数大于目标样本数
    if n_classes > total_n:
        pick_classes = rng.choice(classes, size=total_n, replace=False)
        parts = []
        for c in pick_classes:
            g = df[df[label_col] == c]
            parts.append(g.sample(n=1, random_state=seed))
        sampled = pd.concat(parts, ignore_index=True)
        return sampled

    # 常规情况
    base = total_n // n_classes
    remainder = total_n - base * n_classes  # 先记录但不直接用；我们会先按 base 抽，再补齐

    # 先每类抽 base 或该类全部（若不足）
    taken_list = []
    taken_counts = {}
    for c in classes:
        g = df[df[label_col] == c]
        take = min(base, len(g))
        if take > 0:
            taken = g.sample(n=take, random_state=seed)
        else:
            taken = g.iloc[0:0]  # 空
        taken_list.append(taken)
        taken_counts[c] = len(taken)
    sampled = pd.concat(taken_list, ignore_index=True)

    # 计算还需补多少
    need = total_n - len(sampled)
    if need <= 0:
        return sampled.iloc[:total_n, :]

    # 构造“可补充池”：每类剩余容量 = 该类总量 - 已取
    pool_indices = []
    for c in classes:
        g = df[df[label_col] == c]
        already = taken_counts.get(c, 0)
        cap = len(g) - already
        if cap > 0:
            remain_idx = g.index.difference(sampled.index)
            pool_indices.extend(remain_idx.tolist())

    if len(pool_indices) == 0:
        # 没有可补充的，提前返回（可能总量本就 < total_n）
        return sampled

    # 从全体可补充池随机补到 total_n
    if need > len(pool_indices):
        need = len(pool_indices)
    extra_idx = rng.choice(pool_indices, size=need, replace=False)
    extra = df.loc[extra_idx]
    sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Build balanced 20k EHR report csv with lowercase & shuffle")
    ap.add_argument("--train_csv", default="data/diagnoses.csv",
                    help="path to mimic_diagnosis_icd11(train).csv")
    ap.add_argument("--persona_csv", default="persona.csv", help="path to persona.csv")
    ap.add_argument("--discharge", default="mimic/discharge.csv.gzv",
                    help="path to MIMIC-IV-Note discharge file (default tries .gzv, falls back to .gz)")
    ap.add_argument("--output", default="ehr_report(train)_balanced20k.csv", help="output CSV path")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for sampling/shuffling")
    ap.add_argument("--total_n", type=int, default=20000, help="target total sample size")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1) 读取报告（并先转小写用于分组规范化）
    rep_cols = ["subject_id", "hadm_id", "report", "icd11_title"]
    reports = safe_read_csv(args.train_csv, usecols=rep_cols)
    missing = [c for c in rep_cols if c not in reports.columns]
    if missing:
        raise KeyError(f"{args.train_csv} 缺少列：{missing}")
    reports = coerce_ids(reports)

    # 统一小写（仅字符串）
    if "icd11_title" in reports.columns:
        reports["icd11_title"] = reports["icd11_title"].astype(str).str.lower()
    if "report" in reports.columns:
        reports["report"] = reports["report"].astype(str).str.lower()

    # === 新增：按 icd11_title 分层均衡抽样 20k ===
    reports_bal = stratified_balanced_sample(
        reports,
        label_col="icd11_title",
        total_n=args.total_n,
        seed=args.seed
    )

    # 2) 读取 persona（并小写字符串字段）
    per_cols = ["subject_id", "hadm_id"] + PERSONA_KEYS
    persona = safe_read_csv(args.persona_csv, usecols=per_cols)
    persona = coerce_ids(persona).drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")
    # 将 persona 中的字符串列小写
    for c in PERSONA_KEYS:
        if c in persona.columns and persona[c].dtype == object:
            persona[c] = persona[c].astype(str).str.lower()

    # 合并报告与 persona（左连接）
    df = reports_bal.merge(persona, on=["subject_id", "hadm_id"], how="left")

    # persona 序列化（保证小写）
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

    # 5) 统一将输出中的字符串列小写（report/text 已在各自环节处理；此处稳妥再降一遍）
    for col in ["report", "icd11_title", "text"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    # 6) 输出前随机打乱
    out_cols = ["subject_id", "hadm_id", "icd11_title", "report", "persona", "text"]
    out = coerce_ids(df[out_cols].copy())

    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}  shape={out.shape}")
    print("类分布（前10行）：")
    print(out["icd11_title"].value_counts())

if __name__ == "__main__":
    main()
