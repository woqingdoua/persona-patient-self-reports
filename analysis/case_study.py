import os
os.environ["MPLBACKEND"] = "Agg"  # 避免 PyCharm InterAgg 冲突；如需交互显示可改为 MacOSX/Qt5Agg
import matplotlib as mpl
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= 路径配置 =========
csv_gpt5 = "data/ehr_subset_10k_gpt5_mini.csv"
csv_gpt4o = "data/ehr_subset_10k_gpt4o_mini.csv"
out_path  = "data/fk_edu_gpt4o_vs_gpt5_lines.png"


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ========= 通用函数 =========
def parse_persona(s):
    """把形如 {""k"": ""v""} 的字符串修正为合法 JSON 再 loads。"""
    if pd.isna(s):
        return {}
    s = str(s).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    s = s.replace('""', '"')
    try:
        return json.loads(s)
    except Exception:
        s2 = re.sub(r'(?<!\\)"\s*:\s*"', '":"', s)
        s2 = s2.replace('\\"', '"')
        return json.loads(s2)

_vowel_re = re.compile(r"[aeiouyAEIOUY]+")

def count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    parts = [p for p in parts if p.strip()]
    return max(1, len(parts))

def tokenize_words(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)

def count_syllables_in_word(word: str) -> int:
    w = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not w:
        return 0
    if w.endswith("e"):
        w = w[:-1]
    sylls = len(_vowel_re.findall(w))
    return max(1, sylls)

def fk_grade(text: str) -> float:
    """Flesch–Kincaid Grade = 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59"""
    if not isinstance(text, str):
        return float("nan")
    words = tokenize_words(text)
    n_words = len(words)
    if n_words == 0:
        return float("nan")
    n_sents = count_sentences(text)
    n_sylls = sum(count_syllables_in_word(w) for w in words)
    return 0.39 * (n_words / n_sents) + 11.8 * (n_sylls / n_words) - 15.59

# 教育类别规范化到固定 6 类（并按该顺序出图）
edu_order = ["primary", "middle", "high", "college", "master", "phd"]
edu_alias = {
    # primary
    "elem": "primary", "elementary": "primary", "primary school": "primary",
    # middle
    "middle school": "middle", "junior high": "middle", "middle-school": "middle",
    # high
    "hs": "high", "highschool": "high", "high school": "high", "secondary": "high",
    # college / bachelor
    "bachelor": "college", "bachelors": "college", "ba": "college", "bs": "college",
    "undergrad": "college", "college degree": "college",
    # master
    "masters": "master", "m.sc": "master", "msc": "master", "ms": "master",
    "ma": "master", "graduate": "master", "mres": "master",
    # phd
    "ph.d": "phd", "phd.": "phd", "doctorate": "phd", "dphil": "phd"
}
def normalize_edu(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = s.replace(".", "")
    s = s.replace("'", "")
    s = re.sub(r"\s+", " ", s)
    return edu_alias.get(s, s)

def load_and_aggregate(csv_path: str, report_col: str):
    """读取 csv，解析 persona，提取 education_level，计算 FK，并按固定 6 类聚合均值。"""
    df = pd.read_csv(csv_path)
    needed = [report_col, "persona"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")
    df = df[needed].dropna(subset=[report_col, "persona"]).copy()

    persona_parsed = df["persona"].apply(parse_persona)
    df_p = pd.json_normalize(persona_parsed)
    if "education_level" not in df_p.columns:
        df_p["education_level"] = None
    df["education_level"] = df_p["education_level"].astype("string")

    df["education_level_std"] = df["education_level"].apply(normalize_edu)
    df["fk_grade"] = df[report_col].apply(fk_grade)
    df = df.dropna(subset=["fk_grade"]).copy()

    df = df[df["education_level_std"].isin(edu_order)].copy()
    stats = (
        df.groupby("education_level_std", observed=True)["fk_grade"]
          .mean()
          .reindex(edu_order)  # 固定顺序；即使缺类也能占位
    )
    return stats

def main():
    # ========= 分别加载两份数据并聚合 =========
    edu_stats_gpt4o = load_and_aggregate(csv_gpt4o, "report_gpt4o_mini")
    edu_stats_gpt5  = load_and_aggregate(csv_gpt5,  "report_gpt5_mini")

    # ========= 单图双折线（颜色：#3b5b92 和 #d95f02） =========
    x = np.arange(len(edu_order))
    y4o = edu_stats_gpt4o.values.astype(float)
    y5  = edu_stats_gpt5.values.astype(float)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y4o, marker="o", linewidth=2, color="#3b5b92", label="GPT-4o-mini")
    plt.plot(x, y5,  marker="s", linewidth=2, color="#d95f02", label="GPT-5-mini")

    plt.xticks(x, edu_order)
    plt.xlabel("Education Level")
    plt.ylabel("Mean Flesch–Kincaid Grade")
    #plt.title("Flesch–Kincaid Grade by Education Level")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    mpl.rcParams['font.size'] = 14
    plt.tight_layout()

    # 如需交互显示，把上面的 Agg 后端切为:
    #   import matplotlib; matplotlib.use("MacOSX")  # 或 "Qt5Agg"
    # 并将最后两行改为：
    # plt.tight_layout(); plt.show()


    plt.savefig("data/case_study.png", dpi=200, bbox_inches="tight")
    plt.savefig("data/case_study.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
