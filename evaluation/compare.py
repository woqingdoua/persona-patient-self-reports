#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, random, shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

COMPARE_PROMPT = """You are comparing two first-person patient self-reports. Use ONLY the persona and the PATIENT-STATED EHR excerpt as ground truth. Decide which report is better overall.

Inputs
- Persona (JSON): {persona_json}
- EHR (patient-stated only):
{ehr_text}

Candidates
- Report1:
{report1}
- Report2:
{report2}

Instructions
- Prefer the report that better matches the EHR’s patient-stated facts, sounds natural in a first-person patient voice, and fits the persona’s tone.
- If both are clearly similar in quality, choose "tie".
- Return JSON only; no explanations.

Output (JSON only)
{{
  "winner": 0 | 1 | "tie"
}}
"""

def build_prompt(persona: str, ehr: str, rep1: str, rep2: str) -> str:
    return COMPARE_PROMPT.format(
        persona_json = persona if isinstance(persona, str) else "",
        ehr_text     = ehr if isinstance(ehr, str) else "",
        report1      = rep1 if isinstance(rep1, str) else "",
        report2      = rep2 if isinstance(rep2, str) else "",
    )

def get_client(api_key: str, timeout: int = 60) -> OpenAI:
    proxy = os.environ.get("HTTPS_PROXY")
    http_client = httpx.Client(proxy=proxy, timeout=timeout) if proxy else httpx.Client(timeout=timeout)
    return OpenAI(api_key=api_key, http_client=http_client, timeout=timeout)

def _strip_fences(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    return s

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def judge_once(client: OpenAI, model: str, prompt: str) -> dict:
    # Preferred: Responses API
    try:
        resp = client.responses.create(
            model=model,
            max_output_tokens=120,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "Return JSON only; no explanations."}]},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
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
        return json.loads(_strip_fences(text or ""))
    except Exception:
        pass

    # Fallback: Chat Completions
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only; no explanations."},
            {"role": "user",   "content": prompt},
        ],
    )
    return json.loads(_strip_fences(resp2.choices[0].message.content or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ehr_subset_10k_gpt.csv")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--api_key",
                    default=os.getenv("OPENAI_API_KEY", ""),
                    help="OpenAI API key (or set OPENAI_API_KEY).")

    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Export it or pass --api_key.")
    random.seed(args.seed)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)
    print(f"[start] Loaded {in_path} rows={len(df)}")

    # schema check
    for col in ["persona", "text", "report_gpt4o_mini", "report_gpt5_mini"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # ensure only cmp_winner column
    if "cmp_winner" not in df.columns:
        df["cmp_winner"] = pd.NA

    client = get_client(args.api_key, timeout=args.timeout)
    tmp_path = in_path.with_suffix(in_path.suffix + ".tmp")

    pbar = tqdm(total=len(df), desc=f"Compare → {args.model}")
    processed = 0
    errors = 0

    for i in range(len(df)):
        persona = str(df.at[i, "persona"])
        ehr     = str(df.at[i, "text"])
        r4o     = str(df.at[i, "report_gpt4o_mini"]) if pd.notna(df.at[i, "report_gpt4o_mini"]) else ""
        r5m     = str(df.at[i, "report_gpt5_mini"])  if pd.notna(df.at[i, "report_gpt5_mini"])  else ""

        # skip if missing either
        if not r4o.strip() or not r5m.strip():
            df.at[i, "cmp_winner"] = "tie"
            processed += 1
            pbar.update(1)
            if processed % args.save_every == 0:
                df.to_csv(tmp_path, index=False)
            continue

        # randomize order
        if random.random() < 0.5:
            rep1, rep2 = r4o, r5m
            winner_map = {0: "gpt4o_mini", 1: "gpt5_mini"}
        else:
            rep1, rep2 = r5m, r4o
            winner_map = {0: "gpt5_mini", 1: "gpt4o_mini"}

        prompt = build_prompt(persona, ehr, rep1, rep2)

        try:
            out = judge_once(client, args.model, prompt)
            w = out.get("winner", "tie")
            if isinstance(w, str) and w.lower() == "tie":
                df.at[i, "cmp_winner"] = "tie"
            else:
                try:
                    df.at[i, "cmp_winner"] = winner_map.get(int(w), "tie")
                except Exception:
                    df.at[i, "cmp_winner"] = "tie"
        except Exception:
            errors += 1
            df.at[i, "cmp_winner"] = "tie"

        processed += 1
        pbar.update(1)

        if processed % args.save_every == 0:
            df.to_csv(tmp_path, index=False)
            print(f"[checkpoint] saved {processed} → {tmp_path} (errors={errors})")

    pbar.close()

    # final save: atomic replace + .bak
    df.to_csv(tmp_path, index=False)
    shutil.copyfile(in_path, str(in_path) + ".bak")
    os.replace(tmp_path, in_path)
    print(f"[done] wrote cmp_winner to {in_path} (errors={errors}). Backup: {in_path}.bak")

if __name__ == "__main__":
    main()
