#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute linguistic statistics for three columns in result/ehr_subset_10k_gpt.csv:
    - report
    - report_gpt5_mini
    - report_gpt4o_mini

Per-text metrics:
- avg_length_tokens
- avg_sentences
- avg_sentence_length
- vocabulary_size
- readability (Flesch–Kincaid)
- content_word_density
- lexical_diversity

Output:
- stat.csv : column-level mean of the above metrics
"""

import pandas as pd
import spacy
import nltk
import textstat
from tqdm.auto import tqdm

nlp = None  # Loaded only by the explicitly invoked analysis workflow.

CSV_PATH = "result/ehr_subset_10k_gpt.csv"

# 需要统计的三列
TARGET_COLS = ["report", "report_gpt5_mini", "report_gpt4o_mini"]

def analyze_text(text: str):
    """对单条文本做语言学统计；空值/非字符串返回全 0。"""
    zero = {
        "avg_length_tokens": 0.0,
        "avg_sentences": 0.0,
        "avg_sentence_length": 0.0,
        "vocabulary_size": 0.0,
        "readability": 0.0,
        "content_word_density": 0.0,
        "lexical_diversity": 0.0,
        "adj_ratio": 0.0,
        "adv_ratio": 0.0,
        "verb_ratio": 0.0,
        "noun_ratio": 0.0,
    }
    if not isinstance(text, str) or not text.strip():
        return zero

    doc = nlp(text)
    words = [t for t in doc if t.is_alpha]          # 只统计字母词，与原口径一致
    sents = list(doc.sents)
    n_words = len(words)

    if n_words == 0:
        out = zero.copy()
        out["avg_sentences"] = float(len(sents))
        return out

    avg_length_tokens   = float(n_words)
    avg_sentences       = float(len(sents)) if sents else 0.0
    avg_sentence_length = (n_words / len(sents)) if sents else 0.0
    vocabulary_size     = float(len(set(w.text for w in words)))
    readability         = float(textstat.flesch_kincaid_grade(text))

    # 逐类计数
    adj_cnt  = sum(1 for t in words if t.pos_ == "ADJ")
    adv_cnt  = sum(1 for t in words if t.pos_ == "ADV")
    verb_cnt = sum(1 for t in words if t.pos_ == "VERB")
    noun_cnt = sum(1 for t in words if t.pos_ == "NOUN")

    adj_ratio  = adj_cnt  / n_words
    adv_ratio  = adv_cnt  / n_words
    verb_ratio = verb_cnt / n_words
    noun_ratio = noun_cnt / n_words

    content_word_density = (adj_cnt + adv_cnt + verb_cnt + noun_cnt) / n_words
    lexical_diversity    = len(set(w.text for w in words)) / n_words

    return {
        "avg_length_tokens": avg_length_tokens,
        "avg_sentences": avg_sentences,
        "avg_sentence_length": avg_sentence_length,
        "vocabulary_size": vocabulary_size,
        "readability": readability,
        "content_word_density": content_word_density,
        "lexical_diversity": lexical_diversity,
        "adj_ratio": adj_ratio,
        "adv_ratio": adv_ratio,
        "verb_ratio": verb_ratio,
        "noun_ratio": noun_ratio,
    }


def main():
    global nlp
    tqdm.pandas()
    # punkt (首跑缺失时下载)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    # spaCy 英文模型（未安装：python -m spacy download en_core_web_sm）
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(CSV_PATH)

    results = []
    for col in TARGET_COLS:
        if col not in df.columns:
            print(f"[warn] Column '{col}' not found, skipped.")
            continue
        print(f"[processing] {col}")
        metrics = df[col].progress_apply(analyze_text)
        metrics_df = pd.DataFrame(list(metrics))
        metrics_df["column"] = col
        results.append(metrics_df)

    if not results:
        print("[error] No target columns were processed.")
        return

    full_metrics = pd.concat(results, ignore_index=True)

    # 分列聚合（保留 column 列，方便写出）
    summary = full_metrics.groupby("column", as_index=False).mean(numeric_only=True)

    out_path = "stat.csv"
    summary.to_csv(out_path, index=False)
    print(f"[done] Saved aggregated stats → {out_path}")
    print(summary.round(3))

if __name__ == "__main__":
    main()
