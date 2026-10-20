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
INPUT_CSV = "persona_self_report/self_report_sample_100.csv"
OUTPUT_CSV = "persona_self_report/self_report_sample_100_grade.csv"
MODEL = "gpt-5"
MAX_RETRY = 3
TIMEOUT = 60
N_ROWS = 50

# ---- Prompt ----
TASK_PROMPT = """
**Task**
Grade a patient’s self-report generated from `persona_json` (patient info) and `facts_json` (EHR-derived facts).

**What to use**

* Use only `persona_json`, `facts_json`, and the provided `narrative`.
* Treat `facts_json.positive_symptoms` and `facts_json.negative_symptoms` as the ground truth symptom sets.
* If a symptom set is empty, its score defaults to **1.0**.

**Scoring rules (0–1 each)**

1. **Symptom Coverage**
   **Definition:** Does the narrative clearly include all `positive_symptoms`? Synonyms or closely related descriptions count as coverage.
   **Computation:** (# of positive symptoms expressed in the narrative) ÷ (total # of `positive_symptoms`).

2. **Negative Symptom Compliance**
   **Definition:** Does the narrative explicitly deny all `negative_symptoms`? Synonyms or closely related descriptions of those symptoms count toward compliance.
   **Computation:** (# of negative symptoms explicitly denied in the narrative) ÷ (total # of `negative_symptoms`).

3. **Completeness**
   Definition: Does the narrative cover the six elements? onset/time course; main symptoms; ≥1 trigger or reliever; daily-life impact; self-care/OTC; reason for seeking care.
   Computation: (# elements present) ÷ 6.

**Output constraints**

* Return **only** a JSON object with numeric scores between **0** and **1**, rounded to **two decimals**.
* No extra text.

**Return format**

```json
{
  "Symptom Coverage": 0.00,
  "Negative Symptom Compliance": 0.00,
  "Completeness": 0.00
}
```

**Inputs**

* `persona_json:` …
* `facts_json:` …
* `narrative:` …
"""
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
