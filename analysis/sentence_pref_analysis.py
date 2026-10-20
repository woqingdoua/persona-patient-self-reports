#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

CSV_PATH = "data/ehr_subset_10k_gpt.csv"
OUT_DIR  = "data"
os.makedirs(OUT_DIR, exist_ok=True)

GEN_4O, GEN_5M = "report_gpt4o_mini", "report_gpt5_mini"
WIN_COL = {"GPT-4.1": "cmp_winner", "Claude 3.7": "cmp_winner2"}
COLOR   = {"GPT-4.1": "#3b5b92", "Claude 3.7": "#d95f02"}

sent_split = re.compile(r'[.!?]+(?=\s|$)')
plt.rcParams['font.size'] = 16
def split_sents(t):
    if not isinstance(t, str): return []
    parts = sent_split.split(t.strip())
    return [p.strip() for p in parts if p and p.strip()]

def wc(t):
    if not isinstance(t, str): return 0
    return len(re.findall(r"\b[\w'-]+\b", t))

def text_stats(t):
    s = split_sents(t)
    sc = len(s)
    tw = wc(t)
    return pd.Series({
        "words": tw,
        "sents": sc,
        "avglen": (tw / sc) if sc > 0 else (float(tw) if tw>0 else 0.0),
    })

def canon_winner(v):
    if not isinstance(v, str): return "tie"
    s = v.strip().lower()
    if s in {"4o","4o-mini","gpt-4o-mini","gpt4o_mini","gpt4o"}: return "4o"
    if s in {"5m","5-mini","gpt-5-mini","gpt5_mini","gpt5"}:   return "5m"
    if s in {"tie","equal","same"}: return "tie"
    if "4o" in s: return "4o"
    if "5"  in s: return "5m"
    return "tie"

def prepare_dataframe():
    df = pd.read_csv(CSV_PATH)
    if GEN_4O not in df or GEN_5M not in df:
        raise RuntimeError("缺少文本列 gpt4o_mini / gpt5_mini")

    # 文本统计
    stats4 = df[GEN_4O].apply(text_stats)
    stats5 = df[GEN_5M].apply(text_stats)
    df = pd.concat([df, stats4.add_suffix("_4o"), stats5.add_suffix("_5m")], axis=1)

    # 差分（5m - 4o）
    df["d_sents"]  = df["sents_5m"]  - df["sents_4o"]
    df["d_avglen"] = df["avglen_5m"] - df["avglen_4o"]

    # 为两个 grader 生成0/1 选择变量（去掉 tie）
    frames = []
    for grader, col in WIN_COL.items():
        if col not in df: raise RuntimeError(f"缺少 {col}")
        w = df[col].map(canon_winner)
        ch = np.where(w=="5m", 1.0, np.where(w=="4o", 0.0, np.nan))
        tmp = df.copy()
        tmp["grader"] = grader
        tmp["choose5m"] = ch
        frames.append(tmp)
    long = pd.concat(frames, ignore_index=True)
    # 去掉 tie
    long = long[np.isfinite(long["choose5m"])]
    return long

def partial_curve(subdf, xcol, holdcol, n_pts=60, n_boot=400):
    """
    偏效应曲线：P(选5m) ~ x + hold，在 hold 固定为中位数时画 x 的曲线。
    返回 dict: {grader: dict(xs, ph, lo, hi, OR, OR_CI, color)}
    """
    out = {}
    for grader, color in COLOR.items():
        gdf = subdf[subdf["grader"] == grader]
        if gdf.empty:
            continue
        y = gdf["choose5m"].astype(int).values

        # 标准化（便于 OR per SD 解释）
        x_std   = (gdf[xcol]   - gdf[xcol].mean())   / (gdf[xcol].std(ddof=1)   + 1e-9)
        hold_std= (gdf[holdcol]- gdf[holdcol].mean())/ (gdf[holdcol].std(ddof=1)+ 1e-9)
        X = np.c_[x_std.values, hold_std.values]

        # 逻辑回归
        if len(np.unique(y)) < 2:
            continue
        clf = LogisticRegression(max_iter=300).fit(X, y)
        beta_x = clf.coef_[0, 0]
        OR_per_SD = float(np.exp(beta_x))

        # 画 x 轴（原始尺度），hold 固定中位数
        xs = np.linspace(gdf[xcol].quantile(0.02), gdf[xcol].quantile(0.98), n_pts)
        xs_std = (xs - gdf[xcol].mean()) / (gdf[xcol].std(ddof=1) + 1e-9)
        hold_med = gdf[holdcol].median()
        hold_std_med = (hold_med - gdf[holdcol].mean()) / (gdf[holdcol].std(ddof=1) + 1e-9)
        grid = np.c_[xs_std, np.full_like(xs_std, hold_std_med)]
        p_hat = clf.predict_proba(grid)[:, 1]

        # bootstrap 置信带 + OR CI
        PH = []
        BETA = []
        for _ in range(n_boot):
            bs = resample(gdf, replace=True, n_samples=len(gdf))
            yb = bs["choose5m"].astype(int).values
            xb_std = (bs[xcol] - gdf[xcol].mean()) / (gdf[xcol].std(ddof=1) + 1e-9)
            hb_std = (bs[holdcol] - gdf[holdcol].mean()) / (gdf[holdcol].std(ddof=1) + 1e-9)
            Xb = np.c_[xb_std.values, hb_std.values]
            m = LogisticRegression(max_iter=300).fit(Xb, yb)
            BETA.append(m.coef_[0, 0])
            PH.append(m.predict_proba(grid)[:, 1])
        PH = np.vstack(PH)
        lo, hi = np.percentile(PH, [2.5, 97.5], axis=0)
        OR_CI = tuple(np.exp(np.percentile(BETA, [2.5, 97.5])))

        out[grader] = dict(xs=xs, ph=p_hat, lo=lo, hi=hi,
                           OR=OR_per_SD, OR_CI=OR_CI, color=color)
    return out

def small_density(ax, data, height=0.08, bins=30, color="0.8"):
    """子图底部薄条直方图，显示自变量分布"""
    vals = pd.Series(data).dropna().values
    if len(vals) < 10:
        return
    bins = np.linspace(np.percentile(vals,2), np.percentile(vals,98), bins)
    hist, edges = np.histogram(vals, bins=bins, density=True)
    hist = hist / (hist.max() + 1e-9) * height
    centers = 0.5*(edges[:-1] + edges[1:])
    ax.fill_between(centers, 0, hist, color=color, alpha=0.6, step="mid", zorder=1)

def main():
    df = prepare_dataframe()

    plt.rcParams.update({"figure.figsize": (12, 4.6), "axes.grid": True})
    fig, axes = plt.subplots(1, 2, sharey=True)

    panels = [
        dict(x="d_sents",  hold="d_avglen",
             xlabel="Δ #sentences \n (GPT-5-mini − GPT-4o-mini)"),
        dict(x="d_avglen", hold="d_sents",
             xlabel="Δ avg. sentence length \n (words/sent, GPT-5-mini − GPT-4o-mini)")
    ]

    for ax, P in zip(axes, panels):
        curves = partial_curve(df, xcol=P["x"], holdcol=P["hold"], n_pts=60, n_boot=400)

        # 无偏好灰带
        ax.axhspan(0.45, 0.55, color="0.90", alpha=0.7, lw=0, zorder=0)
        ax.axhline(0.5, ls="--", lw=1, color="gray")
        ax.axvline(0,   ls=":",  lw=1, color="gray")

        # 两条偏效应曲线 + 置信带
        for grader in ["GPT-4.1", "Claude 3.7"]:
            if grader not in curves:
                continue
            c = curves[grader]
            ax.fill_between(c["xs"], c["lo"], c["hi"], color=c["color"], alpha=0.15)
            ax.plot(c["xs"], c["ph"], color=c["color"], lw=2.2,
                    label=f"{grader}  OR/SD≈{c['OR']:.2f} [{c['OR_CI'][0]:.2f}, {c['OR_CI'][1]:.2f}]")

        # 自变量分布（合并两 grader）
        small_density(ax, df[P["x"]].values, height=0.08)

        ax.set_xlabel(P["xlabel"])
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, ls=":", alpha=0.35)

    axes[0].set_ylabel("P(grader picks GPT-5-mini)")
    fig.legend(ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.55, 1.02))
    plt.rcParams['font.size'] = 16
    fig.suptitle("", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    png = os.path.join(OUT_DIR, "drivers_sentence_number_and_length_partial_effects.png")
    pdf = os.path.join(OUT_DIR, "drivers_sentence_number_and_length_partial_effects.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved → {png} / {pdf}")

if __name__ == "__main__":
    main()
