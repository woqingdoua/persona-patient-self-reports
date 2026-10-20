#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multithreaded generation of first-person self-reports from persona+EHR.
- Uses GPT-4o-mini by default.
- Results saved in a new column 'report3'.
- Checkpoints written to tmp2 (default <output or input>.tmp2)
"""

import os, json, argparse, math, time, threading, random
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
from openai import OpenAI

# -------------------- Guardrails (A 版本 persona-aware prompt) --------------------
SYSTEM_MSG = (
    "You are a careful clinical writing assistant. Write exactly one paragraph in the first person ('I…') "
    "using only what the patient stated at presentation. Do NOT include exam findings, vitals interpretation, "
    "labs/imaging, inpatient course, diagnoses, treatment plans, or past history unless explicitly stated today. "
    "Keep explicit denials that the patient stated, but present them briefly and naturally (see rules below). "
    "No headings, no lists, no added facts."
)

PROMPT_TEMPLATE = """Rewrite the patient’s self-report as a natural, conversational first-person paragraph. 
Adapt tone and word choice to the persona below, while strictly staying within patient-stated information. 
Return ONE paragraph in English.

- Persona (JSON):
{persona_json}

- EHR excerpt (patient-stated only):
\"\"\"
{text}
\"\"\"

Denial style rules (to avoid a dense checklist):
- Include ONLY denials explicitly stated in the EHR and relevant to the main concerns.
- Group denials by system and compress into at most TWO short clauses (e.g., “I’m not having fever or chills, and no chest pain or trouble breathing.”).
- Prefer plain connectors (“I’m not having…”, “No…”, “I haven’t noticed…”) over repetitive “I deny…”.
- Place denials once after describing the positive symptoms; do NOT list every possible negative.
- Keep the total denial segment brief (≈15–25 words).

HPI/PMH inclusion rules (patient-stated only):
- HPI focus: Start with what’s happening TODAY or over the RECENT period as the patient told it (onset, course, what changed, why they came). Keep a clear, simple timeline (“today”, “yesterday”, “over the past two days”).
- PMH gating: Mention past conditions, prior procedures, or long-term meds ONLY if the patient explicitly said them TODAY as part of their story (e.g., “I had a kidney transplant and I’m on anti-rejection meds”). Otherwise omit.
- No clinician voice: Do not explain mechanisms, interpret labs/imaging, or suggest diagnoses/plans. Do not summarize ED/ward events or treatments unless the patient said them as part of their presentation story.
- Medication/procedure mentions: Allowed only if the patient stated them today and they are directly tied to why they came in (e.g., “the clinic started a medicine to help my appetite”).
- Temporal anchors: Prefer patient-time anchors over chart-time (e.g., “for the last few weeks”, “this morning”, “after the clinic visit”); avoid calendar dates unless the patient gave one.
- Brevity & relevance: Keep PMH lines ultra-brief and only when they clarify the current complaint; otherwise stay with HPI.


Style constraints (derived from persona; do not add facts):
- Education: match vocabulary/sentence length (high/college/master’s → concise but fluent; middle → simpler words).
- drg_severity: higher severity → tighter, more focused sentences; lower severity → slightly fuller phrasing.
- Insurance: shape tone, not content; don’t mention insurance unless the patient said it.
  - Private → calm, organized, confident cadence.
  - Medicare → plain, steady, matter-of-fact with clear time anchors.
  - Medicaid/Uninsured → slightly urgent, simpler sentences; avoid jargon.
  - If insurance not given → ignore and default to age/education cues.
- admission_location / admission_type (only if stated in EHR):
  - Emergency → tighter, more focused cadence; minimal elaboration.
  - Non-urgent/scheduled/clinic referral → slightly fuller phrasing using only patient-stated context.
- gender & anchor_age: match voice and maturity; avoid slang; age-appropriate simplicity if very old.
- Language: keep English, allow subtle phrasing influenced by background (no new facts).
- marital_status: minimal day-to-day flavor consistent with background (no new medical facts).
"""


def build_prompt(persona_str: str, ehr_text: str) -> str:
    persona_str = persona_str if isinstance(persona_str, str) else ""
    ehr_text = ehr_text if isinstance(ehr_text, str) else ""
    try:
        pj = json.loads(persona_str)
        persona_json = json.dumps(pj, ensure_ascii=False)
    except Exception:
        persona_json = persona_str
    return PROMPT_TEMPLATE.format(persona_json=persona_json, text=(ehr_text or "").strip())

# -------------------- Thread-local OpenAI client --------------------
_tls = threading.local()

def get_client(api_key: str, timeout: int = 60) -> OpenAI:
    cli = getattr(_tls, "client", None)
    if cli is None:
        proxy_url = os.environ.get("HTTPS_PROXY", None)
        http_client = httpx.Client(proxy=proxy_url) if proxy_url else httpx.Client()
        cli = OpenAI(api_key=api_key, http_client=http_client, timeout=timeout)
        _tls.client = cli
    return cli

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def generate_once(api_key: str, model: str, prompt: str, timeout: int = 60) -> str:
    client = get_client(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,         # 对 4o-mini 通常可用；若遇到报错，可去掉此参数
        max_tokens=350,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
    )
    txt = (resp.choices[0].message.content or "").strip()
    return " ".join(txt.split())

def worker_row(i: int, persona: str, text: str, api_key: str, model: str, timeout: int) -> str:
    time.sleep(random.uniform(0.0, 0.05))
    prompt = build_prompt(persona, text)
    try:
        return generate_once(api_key=api_key, model=model, prompt=prompt, timeout=timeout)
    except Exception:
        return ""

# -------------------- IO helpers --------------------
def load_with_resume(input_path: str, tmp_path: str, resume: bool) -> pd.DataFrame:
    if resume and os.path.exists(tmp_path):
        df = pd.read_csv(tmp_path)
        print(f"[resume] Loaded checkpoint: {tmp_path} (rows={len(df)})")
    else:
        df = pd.read_csv(input_path)
        print(f"[start] Loaded input: {input_path} (rows={len(df)})")
    return df

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ehr_subset_10k.csv", help="Input CSV path.")
    ap.add_argument("--output", default="", help="Final output CSV path (default: overwrite input).")
    ap.add_argument("--tmp", default="", help="Temp checkpoint CSV path (default: <output or input>.tmp2)")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id (default: gpt-4o-mini).")
    ap.add_argument("--api_key",
                    default=os.getenv("OPENAI_API_KEY", ""),
                    help="OpenAI API key (or set OPENAI_API_KEY).")

    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout (seconds).")
    ap.add_argument("--only_missing", type=int, default=1, help="Skip rows with non-empty report3.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N rows (0=all).")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool size.")
    ap.add_argument("--save_every", type=int, default=1000, help="Checkpoint batch size.")
    ap.add_argument("--resume", type=int, default=1, help="If 1 and temp exists, resume from it.")
    ap.add_argument("--backup", type=int, default=1, help="If overwriting input, save a .bak before final write.")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not provided. Use --api_key or set OPENAI_API_KEY.")

    out_path = args.output or args.input
    tmp_path = args.tmp or (out_path + ".tmp2")

    df = load_with_resume(args.input, tmp_path, resume=bool(args.resume))

    # Ensure required columns exist
    for col in ["persona", "text"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    if "report_gpt4o_mini" not in df.columns:
        df["report_gpt4o_mini"] = ""

    # Select rows to process
    idxs = list(range(len(df)))
    if args.limit and args.limit > 0:
        idxs = idxs[: args.limit]
    if args.only_missing:
        idxs = [i for i in idxs if not (isinstance(df.at[i, "report_gpt4o_mini"], str) and df.at[i, "report_gpt4o_mini"].strip())]

    if not idxs:
        if os.path.exists(tmp_path):
            if args.backup and not args.output and os.path.exists(args.input):
                pd.read_csv(args.input).to_csv(args.input + ".bak", index=False)
                print(f"[backup] Saved backup: {args.input}.bak")
            os.replace(tmp_path, out_path)
            print(f"[finalize] No rows to process. Wrote final to: {out_path}")
        else:
            df.to_csv(out_path, index=False)
            print(f"[finalize] No rows to process. Saved to: {out_path}")
        return

    # Multithreaded generation
    save_every = max(1, int(args.save_every))
    total = len(idxs)
    pbar = tqdm(total=total, desc="Generating report_gpt4o_mini (multithread)")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for start in range(0, total, save_every):
            batch = idxs[start : start + save_every]
            futures = {}
            for i in batch:
                persona = df.at[i, "persona"]
                ehr_text = df.at[i, "text"]
                fut = ex.submit(worker_row, i, persona, ehr_text, api_key, args.model, args.timeout)
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    df.at[i, "report_gpt4o_mini"] = fut.result()
                except Exception:
                    df.at[i, "report_gpt4o_mini"] = ""
                pbar.update(1)

            df.to_csv(tmp_path, index=False)
            print(f"[checkpoint] Saved {min(start + save_every, total)} / {total} rows to: {tmp_path}")

    pbar.close()
    df.to_csv(tmp_path, index=False)
    print(f"[checkpoint] Saved final checkpoint to: {tmp_path}")

    if args.backup and not args.output and os.path.exists(args.input):
        pd.read_csv(args.input).to_csv(args.input + ".bak", index=False)
        print(f"[backup] Saved backup: {args.input}.bak")

    os.replace(tmp_path, out_path)
    print(f"[done] Wrote final output to: {out_path}")

if __name__ == "__main__":
    main()
