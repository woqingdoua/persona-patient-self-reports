#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample ~10k rows from 'ehr_report(train).csv' maximizing coverage of
all (persona × diagnosis) combinations, without changing the source schema.

- Source is READ-ONLY; output preserves EXACT column order.
- Persona signature uses: marital_status, insurance, admission_location,
  admission_type, gender, education_level, age_bin (<40/40-64/65+).
"""

import argparse
import json
import numpy as np
import pandas as pd

PERSONA_KEYS = [
    "marital_status", "insurance","language", "admission_location", "admission_type",
    "drg_severity", "gender", "anchor_age", "education_level"
]

def parse_persona_signature(s: str) -> str:
    """Build persona signature string from JSON (with age bin)."""
    try:
        d = json.loads(s) if isinstance(s, str) else {}
    except Exception:
        d = {}
    vals = {k: d.get(k, "UNKNOWN") for k in PERSONA_KEYS}
    age = vals.get("anchor_age", "UNKNOWN")
    if isinstance(age, (int, float, np.integer, np.floating)):
        if age < 40: age_bin = "<40"
        elif age < 65: age_bin = "40-64"
        else: age_bin = "65+"
    else:
        age_bin = "UNKNOWN"
    sig_parts = [
        vals["marital_status"], vals["insurance"],
        vals["admission_location"], vals["admission_type"],
        vals["gender"], vals["education_level"], age_bin
    ]
    return " | ".join(map(str, sig_parts))

def greedy_cover(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    """
    3-pass greedy selection to maximize coverage of diseases and persona signatures:
      1) ensure every disease (icd11_title) gets ≥1 sample (prefer unseen persona_sig)
      2) ensure every persona_sig gets ≥1 sample (prefer unseen disease)
      3) fill to quota preferring rows that add a new disease or persona_sig
    Operates on the deduplicated base (unique persona_sig × disease pairs).
    """
    rng = np.random.default_rng(seed)

    base = df.drop_duplicates(subset=["__persona_sig__", "icd11_title"], keep="first").copy()

    selected_idx = []
    seen_dz, seen_sig = set(), set()

    # Pass 1: disease coverage
    for dz, g in base.groupby("icd11_title", sort=False):
        g2 = g[~g["__persona_sig__"].isin(seen_sig)]
        pick = g2.iloc[0] if len(g2) else g.iloc[0]
        selected_idx.append(pick.name)
        seen_dz.add(dz)
        seen_sig.add(pick["__persona_sig__"])
        if len(selected_idx) >= target_n:
            break

    # Pass 2: persona coverage
    if len(selected_idx) < target_n:
        remaining = base.loc[~base.index.isin(selected_idx)]
        for sig, g in remaining.groupby("__persona_sig__", sort=False):
            g2 = g[~g["icd11_title"].isin(seen_dz)]
            pick = g2.iloc[0] if len(g2) else g.iloc[0]
            selected_idx.append(pick.name)
            seen_sig.add(sig)
            seen_dz.add(pick["icd11_title"])
            if len(selected_idx) >= target_n:
                break

    # Pass 3: fill to quota by score (adds new disease/persona)
    if len(selected_idx) < target_n:
        remaining = base.loc[~base.index.isin(selected_idx)].copy()
        remaining["__score__"] = (
            (~remaining["icd11_title"].isin(seen_dz)).astype(int) +
            (~remaining["__persona_sig__"].isin(seen_sig)).astype(int)
        )
        # break ties randomly but reproducibly
        remaining["__rand__"] = rng.random(len(remaining))
        remaining = remaining.sort_values(["__score__", "__rand__"], ascending=[False, True])
        for idx, row in remaining.iterrows():
            selected_idx.append(idx)
            seen_dz.add(row["icd11_title"])
            seen_sig.add(row["__persona_sig__"])
            if len(selected_idx) >= target_n:
                break

    out = base.loc[selected_idx].copy()

    # If unique pairs < target, top up from the original df (still preserving schema)
    if len(out) < target_n:
        remainder = df.loc[~df.index.isin(out.index)]
        need = min(target_n - len(out), len(remainder))
        if need > 0:
            extra = remainder.sample(n=need, random_state=seed)
            out = pd.concat([out, extra], axis=0)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ehr_report(train).csv", help="source CSV (read-only)")
    ap.add_argument("--output", default="ehr_subset_10k.csv", help="output CSV")
    ap.add_argument("--n", type=int, default=10_000, help="target sample size (default 10k)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()
    np.random.seed(args.seed)

    # Load (read-only) and preserve exact schema/order
    df = pd.read_csv(args.input)
    original_cols = df.columns.tolist()

    # Basic checks
    for c in ["persona", "icd11_title"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # Build persona signature (auxiliary column, not saved)
    df["__persona_sig__"] = df["persona"].apply(parse_persona_signature)

    # Run greedy covering selection
    selected = greedy_cover(df, target_n=args.n, seed=args.seed)

    # Write output with EXACT same columns/order as source
    out = selected[original_cols].reset_index(drop=True)
    out.to_csv(args.output, index=False)

    # Simple logs
    print(f"Saved {len(out)} rows to {args.output}")
    print(f"Diseases covered: {out['icd11_title'].nunique()} / {df['icd11_title'].nunique()}")

if __name__ == "__main__":
    main()
