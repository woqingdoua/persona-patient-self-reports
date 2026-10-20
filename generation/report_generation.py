#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multithreaded generation of first-person self-reports from persona+EHR.
- Uses GPT-5-mini by default (Responses API first, fallback to Chat Completions).
- Results saved in column 'report_gpt5_mini'.
- Checkpoints written to tmp2 (default <output or input>.tmp2).
"""

import os, json, argparse, time, threading, random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
from openai import OpenAI

# -------------------- Guardrails (persona-aware) --------------------
SYSTEM_MSG = (
    "You are a careful clinical writing assistant. Write exactly one paragraph in the first person ('I…') "
    "using only what the patient stated at presentation. Do NOT include exam findings, vitals interpretation, "
    "labs/imaging, inpatient course, diagnoses, treatment plans, or past history unless explicitly stated today. "
    "Keep explicit denials that the patient stated, but present them briefly and naturally. "
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
- Group denials by system and compress into at most TWO short clauses.
- Prefer plain connectors (“I’m not having…”, “No…”, “I haven’t noticed…”); avoid repetitive “I deny…”.
- Place denials once after positive symptoms; keep the denial segment brief (~15–25 words).

HPI/PMH inclusion rules (patient-stated only):
- Focus on what’s happening today/recently as the patient told it (onset, course, why they came).
- Mention past conditions/procedures/meds ONLY if the patient stated them today and they clarify the complaint.
- No clinician voice (no mechanisms/interpretation/diagnoses/plans or ED course).
- Use patient-time anchors (“today”, “yesterday”, “over the past two days”); avoid calendar dates unless patient gave one.

Style constraints (derived from persona; do not add facts):
- Education: match vocabulary/sentence length (high/college/master’s → concise but fluent; middle → simpler words).
- drg_severity: higher severity → tighter, more focused sentences; lower severity → slightly fuller phrasing.
- Insurance: shape tone, not content; don’t mention insurance unless the patient said it.
  - Private → calm, organized, confident cadence.
  - Medicare → plain, steady, matter-of-fact with clear time anchors.
  - Medicaid/Uninsured → slightly urgent, simpler sentences; avoid jargon.
- admission_location / admission_type (only if stated in EHR):
  - Emergency → tighter, more focused cadence; minimal elaboration.
  - Non-urgent/scheduled/clinic referral → slightly fuller phrasing using only patient-stated context.
- gender & anchor_age: match voice and maturity; avoid slang; age-appropriate simplicity if very old.
- Language: keep English; subtle phrasing influenced by background is okay (no new facts).
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

def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())

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

# -------------------- LLM call (Responses API first) --------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def generate_once(api_key: str, model: str, prompt: str, timeout: int = 60) -> str:
    client = get_client(api_key=api_key, timeout=timeout)

    # Preferred: Responses API (supports GPT-5 models); do NOT send temperature for models that only support default.
    try:
        resp = client.responses.create(
            model=model,
            max_output_tokens=350,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ],
        )
        text = getattr(resp, "output_text", None)
        if not text:
            # Fallback extraction for structured outputs
            chunks = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            chunks.append(c.text or "")
            text = "".join(chunks)
        if text:
            return _normalize(text)
    except Exception:
        pass

    # Secondary: Responses API with "input" (some SDK variants)
    try:
        resp = client.responses.create(
            model=model,
            max_output_tokens=350,
            input=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt},
            ],
        )
        text = getattr(resp, "output_text", None)
        if not text:
            chunks = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            chunks.append(c.text or "")
            text = "".join(chunks)
        if text:
            return _normalize(text)
    except Exception:
        pass

    # Fallback: Chat Completions (avoid max_tokens/temperature for GPT-5-* to prevent 400 errors)
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt + "\n\nLimit to ONE paragraph, ~120 words."},
        ],
    )
    return _normalize(resp2.choices[0].message.content or "")

def worker_row(i: int, persona: str, text: str, api_key: str, model: str, timeout: int) -> str:
    time.sleep(random.uniform(0.0, 0.05))  # jitter to smooth bursts
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
    ap.add_argument("--input", default="ehr_subset_10k_gpt5.csv", help="Input CSV path.")
    ap.add_argument("--output", default="", help="Final output CSV path (default: overwrite input).")
    ap.add_argument("--tmp", default="", help="Temp checkpoint CSV path (default: <output or input>.tmp2)")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI model id (default: gpt-5-mini).")
    ap.add_argument("--api_key",
                    default=os.getenv("OPENAI_API_KEY", ""),
                    help="OpenAI API key (or set OPENAI_API_KEY).")

    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout (seconds).")
    ap.add_argument("--only_missing", type=int, default=1, help="Skip rows with non-empty report_gpt5_mini.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N rows (0=all).")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool size.")
    ap.add_argument("--save_every", type=int, default=1000, help="Checkpoint batch size.")
    ap.add_argument("--resume", type=int, default=1, help="If 1 and temp exists, resume from it.")
    ap.add_argument("--backup", type=int, default=1, help="If overwriting input, save a .bak before final write.")
    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError("OpenAI API key not provided. Use --api_key or set OPENAI_API_KEY.")

    out_path = args.output or args.input
    tmp_path = args.tmp or (out_path + ".tmp2")

    df = load_with_resume(args.input, tmp_path, resume=bool(args.resume))

    # Ensure required columns exist
    for col in ["persona", "text"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    if "report_gpt5_mini" not in df.columns:
        df["report_gpt5_mini"] = ""

    # Choose rows to process
    idxs = list(range(len(df)))
    if args.limit and args.limit > 0:
        idxs = idxs[: args.limit]
    if args.only_missing:
        idxs = [i for i in idxs if not (isinstance(df.at[i, "report_gpt5_mini"], str) and df.at[i, "report_gpt5_mini"].strip())]

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

    # Multithreaded generation + checkpoints
    save_every = max(1, int(args.save_every))
    total = len(idxs)
    pbar = tqdm(total=total, desc="Generating report_gpt5_mini (multithread)")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for start in range(0, total, save_every):
            batch = idxs[start : start + save_every]
            futures = {}
            for i in batch:
                persona = df.at[i, "persona"]
                ehr_text = df.at[i, "text"]
                fut = ex.submit(worker_row, i, persona, ehr_text, args.api_key, args.model, args.timeout)
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    df.at[i, "report_gpt5_mini"] = fut.result()
                except Exception:
                    df.at[i, "report_gpt5_mini"] = ""
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
