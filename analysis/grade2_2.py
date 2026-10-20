# -*- coding: utf-8 -*-
import os, math, numpy as np, pandas as pd

# 可选：SciPy 用于二项检验与卡方分布（若没有也可运行，p值会置为 NaN/用近似）
try:
    from scipy.stats import binomtest, chi2
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------- 工具函数 ----------
def normalize_label(x):
    """将胜者标签归一化为 {'4o','5m','tie', None}，尽量兼容常见写法。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    alias_4o = {"gpt-4o-mini","gpt4o-mini","4o","4o-mini","gpt-4o mini","gpt4o"}
    alias_5m = {"gpt-5-mini","gpt5-mini","5m","5-mini","gpt-5 mini","gpt5"}
    alias_tie= {"tie","equal","same","draw"}
    if s in alias_4o: return "4o"
    if s in alias_5m: return "5m"
    if s in alias_tie: return "tie"
    if "5" in s and "mini" in s: return "5m"
    if "4o" in s or "4-o" in s or "4 o" in s: return "4o"
    if "tie" in s or "draw" in s or "equal" in s: return "tie"
    return None

def wilson_ci(k, n):
    """Wilson 95%CI for proportion k/n。"""
    if n == 0: return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return (center - half, center + half)

def mcnemar_test(b, c):
    """McNemar（Edwards 连续性校正）。b= A=1,B=0；c= A=0,B=1。"""
    if b + c == 0: return (float("nan"), float("nan"))
    chi2_stat = (abs(b - c) - 1)**2 / (b + c)
    if HAS_SCIPY:
        p = 1 - chi2.cdf(chi2_stat, df=1)
    else:
        p = float("nan")
    return (chi2_stat, p)

def cohens_kappa_from_cm(cm):
    """Cohen's κ（任意 K×K 混淆矩阵）。"""
    n = cm.sum()
    if n == 0: return float("nan")
    po = np.trace(cm) / n
    rows = cm.sum(axis=1); cols = cm.sum(axis=0)
    pe = (rows @ cols) / (n**2)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")

# ---------- 主流程 ----------
CSV = "result/ehr_subset_10k_gpt.csv"
def win_summary(series: pd.Series):
    s = series.dropna()
    n = len(s)
    c4, c5, ct = int((s=="4o").sum()), int((s=="5m").sum()), int((s=="tie").sum())
    n_no_tie = n - ct
    p5_excl   = c5 / n_no_tie if n_no_tie>0 else float("nan")       # 去掉平局
    p5_half   = (c5 + 0.5*ct) / n if n>0 else float("nan")          # 平局计 0.5
    ci_low, ci_high = wilson_ci(c5, n_no_tie) if n_no_tie>0 else (float("nan"), float("nan"))
    if HAS_SCIPY and n_no_tie>0:
        p_binom = binomtest(c5, n_no_tie, p=0.5, alternative="two-sided").pvalue
    else:
        p_binom = float("nan")
    return pd.Series({
        "n_total": n, "n_4o": c4, "n_5m": c5, "n_tie": ct,
        "prop_4o": c4/n if n else float("nan"),
        "prop_5m": c5/n if n else float("nan"),
        "prop_tie": ct/n if n else float("nan"),
        "p5_excl_tie": p5_excl, "p5_excl_tie_CI_low": ci_low, "p5_excl_tie_CI_high": ci_high,
        "p5_half_tie": p5_half, "binom_p_vs_0.5": p_binom,
        "cohen_g": p5_excl - 0.5 if not math.isnan(p5_excl) else float("nan"),
    })

def main():
    assert os.path.exists(CSV), f"找不到文件：{CSV}"

    df = pd.read_csv(CSV)
    assert "cmp_winner" in df.columns and "cmp_winner2" in df.columns, \
           "CSV 需要包含列：cmp_winner（GPT-4.1），cmp_winner2（Claude 3.7）"

    df["w_gpt41"] = df["cmp_winner"].map(normalize_label)
    df["w_cld37"] = df["cmp_winner2"].map(normalize_label)

    sum_gpt41 = win_summary(df["w_gpt41"])
    sum_cld37 = win_summary(df["w_cld37"])
    summary = pd.concat([sum_gpt41, sum_cld37], axis=1)
    summary.columns = ["GPT-4.1 grader", "Claude 3.7 grader"]

    # 评审一致性
    cats = ["4o","5m","tie"]
    cm3 = pd.crosstab(df["w_gpt41"], df["w_cld37"]).reindex(index=cats, columns=cats, fill_value=0).to_numpy()
    kappa_3 = cohens_kappa_from_cm(cm3)

    # 去掉平局的二分类一致性与 McNemar
    mask = df["w_gpt41"].isin(["4o","5m"]) & df["w_cld37"].isin(["4o","5m"])
    a = (df.loc[mask,"w_gpt41"]=="5m").astype(int).to_numpy()
    b = (df.loc[mask,"w_cld37"]=="5m").astype(int).to_numpy()
    b01 = int(((a==1)&(b==0)).sum())  # GPT-4.1 选 5m、Claude 选 4o
    b10 = int(((a==0)&(b==1)).sum())  # GPT-4.1 选 4o、Claude 选 5m
    chi2_stat, mcn_p = mcnemar_test(b01, b10)
    cm2 = np.array([[int(((a==0)&(b==0)).sum()), int(((a==0)&(b==1)).sum())],
                    [int(((a==1)&(b==0)).sum()), int(((a==1)&(b==1)).sum())]])
    kappa_2 = cohens_kappa_from_cm(cm2)

    # 保存输出
    os.makedirs("grade", exist_ok=True)
    summary.to_csv("grade/winner_summary.csv")
    pd.DataFrame(cm3, index=cats, columns=cats).to_csv("grade/winner_confusion_3x3.csv")
    pd.DataFrame(cm2, index=["4o","5m"], columns=["4o","5m"]).to_csv("grade/winner_confusion_2x2.csv")

    print("\n=== Winner summary (per grader) ===")
    print(summary.round(4))
    print("\nKappa (3-class):", round(kappa_3, 3))
    print("Kappa (2-class, ties removed):", round(kappa_2, 3))
    print(f"McNemar: b01={b01}, b10={b10}, chi2={chi2_stat:.3f}, p={mcn_p:.3g}")
    print("\nSaved: grade/winner_summary.csv, winner_confusion_3x3.csv, winner_confusion_2x2.csv")


if __name__ == "__main__":
    main()
