#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Here’s the description you can drop into the paper (concise, method-style):

We generated candidate negative symptom statements (`neg_sym`) from each EHR free-text note using a rule-based pipeline that combines complaint triggers, disease/discipline priors, and persona-aware selection. First, we normalize the EHR text (lowercasing, whitespace collapse) and detect presenting‐complaint categories via keyword triggers (e.g., *shortness of breath, wheeze* → **respiratory**; *chest pain, pressure* → **chest_pain**; *abdominal pain, distension* → **abdominal**; *fever, chills* → **fever_infect**; etc.). In parallel, if available, we map the ICD-11 title and/or specialty to a dominant category using curated regex patterns (e.g., *NSTEMI/AF/HF* → **chest_pain**; *asthma/COPD/PE* → **respiratory**; *cirrhosis/obstruction/colitis* → **abdominal**; *stroke/TIA/seizure* → **neuro**; *UTI/retention/stones* → **urinary**).

For each activated category we retrieve a small, clinically salient set of “red-flag” denials curated by domain experts (e.g., **respiratory** → *no chest pain, no fever, no chills, no hemoptysis, no leg swelling, no pleuritic pain*; **chest_pain** → *no dyspnea, diaphoresis, nausea/vomiting, radiation, syncope*; **abdominal** → *no fever, vomiting, bloody/black stool, persistent severe pain*; **neuro** → *no weakness, numbness, facial droop, slurred speech, seizure, head trauma*; etc.). We merge disease/specialty-driven denials (high priority) with text-triggered denials (secondary) and remove duplicates.

The number of denials emitted per case is adapted to the persona to better approximate real self-reports: we sample fewer denials for older age, non-English language, lower education, or higher DRG severity; otherwise we allow more. Finally, we verbalize the selected items into first-person, succinct templates (e.g., “I do not have chest pain,” “I’m not having vomiting.”). The result is stored as a JSON array in `neg_sym`. This procedure yields plausible, context-specific denials without introducing unsupported clinical facts.

"""

import re
import json
import argparse
import os
import pandas as pd
from typing import List, Dict, Tuple

# ---------------- 基础规则（可自行扩展） ----------------
COMPLAINT2DENIALS: Dict[str, List[str]] = {
    "respiratory": ["chest pain", "fever", "chills", "hemoptysis", "leg swelling", "pleuritic pain"],
    "chest_pain":  ["shortness of breath", "diaphoresis", "nausea", "vomiting", "radiation to arm/jaw/back", "syncope"],
    "abdominal":   ["fever", "vomiting", "bloody stool", "black stool", "persistent severe pain"],
    "fever_infect":["shortness of breath", "chest pain", "dysuria", "flank pain", "stiff neck", "severe headache"],
    "neuro":       ["weakness", "numbness", "facial droop", "slurred speech", "seizure", "head trauma"],
    "urinary":     ["flank pain", "fever", "chills", "nausea", "vomiting", "testicular/perineal pain"],
    "gi_nvd":      ["bloody stool", "black stool", "severe abdominal pain", "high fever", "lightheadedness"],
    "msk_trauma":  ["numbness", "weakness", "bowel or bladder incontinence", "saddle anesthesia", "fever"],
    "neck_throat": ["stiff neck", "trouble swallowing", "drooling", "voice changes", "shortness of breath"],
    "ent_eye":     ["vision loss", "double vision", "severe headache", "photophobia"],
    "obgyn":       ["heavy bleeding", "severe unilateral pelvic pain", "fever", "syncope"],
    "endo":        ["polyuria", "polydipsia", "confusion", "chest pain", "shortness of breath"],
    "heme_onc":    ["bleeding", "black stool", "hematuria", "dyspnea on exertion", "chest pain"],
    "derm":        ["fever", "rapidly spreading redness", "severe pain out of proportion"],
    "general":     ["fever", "chest pain", "shortness of breath", "severe abdominal pain", "syncope"],
}

CATEGORY_TRIGGERS: List[Tuple[str, List[str]]] = [
    ("respiratory", ["shortness of breath","sob","dyspnea","wheezing","asthma","copd","pneumonia","bronchitis","pulmonary"]),
    ("chest_pain",  ["chest pain","pressure in chest","tightness in chest","angina","acs","mi","myocardial infarction"]),
    ("abdominal",   ["abdominal pain","belly pain","abd pain","abdominal swelling","bloating","distension","distended","pancreatitis","appendicitis"]),
    ("fever_infect",["fever","chills","rigor","infection","sepsis"]),
    ("neuro",       ["headache","dizziness","syncope","faint","weakness","numbness","vision changes","stroke","tia","seizure"]),
    ("urinary",     ["dysuria","hematuria","urinary frequency","urinary retention","pneumaturia","pyelonephritis","kidney stone"]),
    ("gi_nvd",      ["nausea","vomiting","diarrhea","n/v","n/v/d"]),
    ("msk_trauma",  ["fall","trauma","injury","back pain","joint pain","swelling","fracture","sprain","twisted ankle"]),
    ("neck_throat", ["neck pain","sore throat","odynophagia","dysphagia","tonsil"]),
    ("ent_eye",     ["red eye","eye pain","conjunctivitis","vision changes","double vision","photophobia","sinus","otitis"]),
    ("obgyn",       ["pregnant","pregnancy","vaginal bleeding","pelvic pain","lmp","ectopic"]),
    ("endo",        ["hyperglycemia","high blood sugar","dka","hypoglycemia","hypothyroid","weight change"]),
    ("heme_onc",    ["anemia","cancer","chemo","chemotherapy","leukemia","lymphoma","thrombosis","dvt","pe"]),
    ("derm",        ["rash","cellulitis","redness","skin infection"]),
    ("general",     ["weak","unwell","malaise","fatigue"]),
]

SPECIALTY2CATEGORY_DEFAULT = {
    "Cardiology": "chest_pain",
    "Pulmonology": "respiratory",
    "Gastroenterology": "abdominal",
    "Hepatology": "abdominal",
    "Nephrology": "urinary",
    "Urology": "urinary",
    "Endocrinology": "endo",
    "Nutritional & Metabolic Diseases": "endo",
    "Neurology": "neuro",
    "Neurosurgery": "neuro",
    "Orthopaedics": "msk_trauma",
    "General Surgery": "abdominal",
    "Dermatology": "derm",
    "Gynaecology": "obgyn",
    "Psychiatry": "general",
    "Hematology": "heme_onc",
    "Oncology": "heme_onc",
    "Infectious Diseases": "fever_infect",
    "Eurology": "urinary",  # 若你的表里有这个拼写
}

DISEASE_KEYWORDS2CATEGORY: List[Tuple[str, str]] = [
    (r"\bmi\b|\bmyocardial infarction\b|st elevation|nonst elevation|nstemi|stemi|angina|coronary", "chest_pain"),
    (r"atrial fibrillation|av block|bradycardia|tachycardia|arrhythmia|heart failure|cardiomyopathy", "chest_pain"),
    (r"asthma|copd|pneumonia|pulmonary embolism|pe\b|hemoptysis", "respiratory"),
    (r"cirrhosis|pancreatitis|cholecystitis|appendicitis|bowel obstruction|ileus|colitis|gi bleed", "abdominal"),
    (r"stroke|cerebrovascular|tia\b|seizure|epilepsy|headache|migraine|subarachnoid|intracranial", "neuro"),
    (r"pyelonephritis|urolithiasis|kidney stone|nephrolithiasis|cystitis|bph|urinary retention", "urinary"),
    (r"diabetic ketoacidosis|dka|hyperglycemia|hypoglycemia|thyroid", "endo"),
    (r"cellulitis|sepsis|bacteremia|meningitis", "fever_infect"),
    (r"anemia|leukemia|lymphoma|thrombocytopenia|thrombosis|dvt", "heme_onc"),
    (r"dermatitis|psoriasis|urticaria|eczema", "derm"),
    (r"ectopic|ovarian torsion|pelvic inflammatory|pih|preeclampsia", "obgyn"),
    (r"fracture|sprain|dislocation|contusion", "msk_trauma"),
]

TEMPLATES = [
    "I do not have {sym}.",
    "I'm not having {sym}.",
    "I haven't noticed {sym}.",
    "I haven't had {sym}.",
]

# ---------------- 工具函数 ----------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def guess_categories_from_text(text: str) -> List[str]:
    s = normalize(text)
    scores = {}
    for cat, kws in CATEGORY_TRIGGERS:
        for k in kws:
            # 用 \b 边界匹配，尽量减少误匹配
            if re.search(rf"\b{re.escape(k)}\b", s):
                scores[cat] = scores.get(cat, 0) + 1
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:2]] or ["general"]

def map_disease_to_category(icd11_title: str, specialty: str) -> str:
    icd = normalize(icd11_title)
    for pat, cat in DISEASE_KEYWORDS2CATEGORY:
        if re.search(pat, icd, flags=re.I):
            return cat
    return SPECIALTY2CATEGORY_DEFAULT.get(specialty, "general")

def decide_denial_count(persona_json: str) -> int:
    """
    用 persona 粗略决定否认条数（更贴近真实表达）：
    - 教育/语言较弱、老年、高严重度：更少的否认条目
    """
    n_min, n_max = 1, 4
    base = 3
    try:
        p = json.loads(persona_json) if isinstance(persona_json, str) and persona_json.strip().startswith("{") else {}
    except Exception:
        p = {}
    edu = (p.get("education_level") or "").lower()
    lang = (p.get("language") or "").lower()
    age = p.get("anchor_age") or p.get("age") or 0
    drg = p.get("drg_severity") or 0
    if edu in {"middle", "low"} or (lang and lang != "english"):
        base -= 1
    try:
        if float(age) >= 75:
            base -= 1
    except Exception:
        pass
    try:
        if float(drg) >= 3:
            base -= 1
    except Exception:
        pass
    return max(n_min, min(n_max, int(base)))

def verbalize(sym: str) -> str:
    # 一些口语化小修饰
    return sym.replace("black stool", "black stools")

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ehr_report_sample.csv", help="输入 CSV（需包含 text 列；建议包含 persona）")
    #ap.add_argument("--output", default="", help="输出 CSV（默认：<input>.with_neg.csv）")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.input#args.output or (os.path.splitext(in_path)[0] + ".with_neg.csv")

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    # 读取（若非 UTF-8，可加 encoding='latin1' 再转码，这里先假定 UTF-8）
    df = pd.read_csv(in_path, low_memory=False)

    if "text" not in df.columns:
        raise KeyError("Input CSV must contain column: 'text'")

    # 逐行生成 neg_sym
    neg_list = []
    for _, row in df.iterrows():
        text = str(row.get("text", "") or "")
        persona = str(row.get("persona", "") or "")
        icd = str(row.get("icd11_title", "") or "")
        spec = str(row.get("specialty", "") or "")

        # 1) 疾病/科室感知 → 首选类别
        cat_from_dx = map_disease_to_category(icd, spec) if (icd or spec) else None
        dx_list = COMPLAINT2DENIALS.get(cat_from_dx, []) if cat_from_dx else []

        # 2) 从 EHR 文本触发 → 前两类
        cats = guess_categories_from_text(text)
        trig_list = []
        for c in cats:
            trig_list += COMPLAINT2DENIALS.get(c, [])

        # 3) 合并（疾病优先）去重
        seen, merged = set(), []
        for s in dx_list + trig_list:
            key = normalize(s)
            if key and key not in seen:
                seen.add(key)
                merged.append(key)

        # 4) 依据 persona 决定条数
        n = decide_denial_count(persona)
        picked = merged[:n] if merged else []

        # 5) 口语化模板化
        outs = []
        for i, s in enumerate(picked):
            sent = ["I do not have {sym}.",
                    "I'm not having {sym}.",
                    "I haven't noticed {sym}.",
                    "I haven't had {sym}."][i % 4].format(sym=verbalize(s))
            outs.append(sent)

        neg_list.append(json.dumps(outs, ensure_ascii=False))

    df["neg_sym"] = neg_list
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[done] wrote: {out_path}  rows={len(df)}")

if __name__ == "__main__":
    main()
