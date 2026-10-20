#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time
import pandas as pd
from openai import OpenAI
import httpx
from tqdm import tqdm

# Constants
API_KEY = os.getenv("OPENAI_API_KEY", "")

client = None  # Initialized only by the explicitly invoked workflow.


# ===== Config =====
INPUT_CSV = "persona_self_report.csv"
OUTPUT_CSV = "persona_self_report/self_report_sample_50(gpt-4o-mini).csv"
MODEL = "gpt-4o-mini"
MAX_RETRY = 3
TIMEOUT = 60
N_ROWS = 50

# ---- Prompt ----
TASK_PROMPT = (
    "Generate a realistic first-person patient self-report using only persona_json (patient's inforamtion) and facts_json (EHR).\n\n"
    "Rules:\n"
    "1) Mention all positive_symptoms clearly, and **explicitly** state the absence of all negative_symptoms.\n"
    "2) You may describe onset/time course, main symptoms, ≥1 trigger or reliever, daily-life impact, self-care/OTC, and the reason for seeking care.\n"
    "3) Match sentence length and vocabulary to persona_json.education_level.\n"
    "4) Add small, plausible details only if they help diagnosis and are consistent with persona_json and facts_json; do not introduce new clinical facts.\n"
    "5) Do not include tests, labs, imaging, clinician opinions, procedures, or diagnostic labels.\n"
    "6) Output exactly one coherent paragraph in plain English (no headings or lists).\n"
    "7) Include relevant history only if present in facts_json.\n"
)

def make_user_message(persona_json: str, facts_json: str) -> str:
    return f"{TASK_PROMPT}\npersona_json: {persona_json}\nfacts_json: {facts_json}\n"


def call_model(persona_json: str, facts_json: str) -> str:
    user_content = make_user_message(persona_json, facts_json)
    last_err = None
    for _ in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.7,
                max_tokens=350,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
                return text
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    print(f"[warn] generation failed after retries: {last_err}")
    return ""

def main():
    global client
    proxy_url = os.environ.get("HTTPS_PROXY")
    http_client = httpx.Client(proxy=proxy_url) if proxy_url else httpx.Client()
    client = OpenAI(api_key=API_KEY, http_client=http_client)

    df = pd.read_csv(INPUT_CSV).head(N_ROWS).copy()
    if "text" not in df.columns:
        raise KeyError("facts_json must come from 'text' column, but 'text' not found in input CSV.")
    if "persona" not in df.columns:
        raise KeyError("Missing 'persona' column (expected persona_json).")
    df = df[["subject_id", "hadm_id", "persona", "text"]].copy()
    df["persona"] = df["persona"].astype(str)
    df["facts"] = df["text"].astype(str)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        sr = call_model(r["persona"], r["facts"])
        rows.append({
            "subject_id": r["subject_id"],
            "hadm_id": r["hadm_id"],
            "persona": r["persona"],
            "self_report": sr,
            "text": r["facts"]
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved: {OUTPUT_CSV}  rows={len(out_df)}")


if __name__ == "__main__":
    main()
