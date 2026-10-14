"""Figures for probe results.

    python plot_probe.py results/hotspot_nonadj_d2.5.json            # frontiers
    python plot_probe.py results/X.json --heatmap demand_mbps hotspot_layout --metric ES_orC
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {  # fixed categorical order of the reference palette
    "rule": ("#2a78d6", "o", "Rule pair (paper)"),
    "ruleT": ("#eb6834", "s", "Rule pair, MLB dead zone"),
    "ruleES0": ("#1baf7a", "^", "ES rule, CIO = 0"),
    "ruleC": ("#eda100", "D", "ES rule + CIO oracle"),
    "orR": ("#e87ba4", "v", "Oracle ES + MLB rule  (≈ ES-only learning)"),
    "or0": ("#008300", "P", "Oracle ES, CIO = 0"),
    "orC": ("#4a3aa7", "X", "Oracle ES + CIO oracle  (≈ joint bound)"),
}
BLUE = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#e34948", "#f0a3a2", "#f0efec", "#9ec5f4", "#2a78d6"]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def frontiers(table, v_max, path, names=None):
    """Energy saving vs mean V for every family, one panel per config."""
    rows = [r for r in table if names is None or r["config"] in names][:4]
    fig, axes = plt.subplots(1, len(rows), figsize=(max(4.4 * len(rows), 6.0), 4.2), sharey=True,
                             squeeze=False)
    for ax, row in zip(axes[0], rows):
        style(ax)
        for fam, (c, m, lab) in SERIES.items():
            pts = sorted([p for p in row["points"] if p["fam"] == fam], key=lambda p: p["par"])
            if pts:
                ax.plot([p["V"] for p in pts], [100 * p["ES"] for p in pts], color=c, lw=2,
                        marker=m, ms=6, mec="#fcfcfb", mew=1.5, label=lab)
        for k, lab, dy in [("allon", "All-on", 8), ("allon_cio", "All-on + CIO oracle", -12)]:
            p = next((p for p in row["points"] if p["fam"] == k), None)
            if p:
                ax.plot([p["V"]], [100 * p["ES"]], marker="o", ms=7, color=INK2, mfc="none",
                        ls="none")
                ax.annotate(lab, (p["V"], 100 * p["ES"]), textcoords="offset points",
                            xytext=(4, dy), fontsize=7, color=INK2)
        ax.axvline(v_max, color=INK2, lw=1, ls="--")
        ax.text(v_max, ax.get_ylim()[1], " V_max", color=INK2, fontsize=8, va="top")
        ax.set_xlabel("mean service degradation  V̄", color=INK2, fontsize=9)
        ax.set_title(row["config"], color=INK, fontsize=9, loc="left")
    axes[0][0].set_ylabel("energy saving vs All-on  [%]", color=INK2, fontsize=9)
    axes[0][-1].legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper left",
                       bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#f9f9f7")
    plt.close(fig)


def heatmap(table, cfgs, rows_key, cols_key, metric, path, diverging=False):
    """metric (x100) over two swept config keys."""
    R = sorted({cfgs[r["config"]][rows_key] for r in table}, key=str)
    C = sorted({cfgs[r["config"]][cols_key] for r in table}, key=str)
    M = np.full((len(R), len(C)), np.nan)
    for r in table:
        kw = cfgs[r["config"]]
        M[R.index(kw[rows_key]), C.index(kw[cols_key])] = 100 * r.get(metric, np.nan)
    fig, ax = plt.subplots(figsize=(1.1 * len(C) + 2, 0.6 * len(R) + 1.6))
    if diverging:
        lim = max(np.nanmax(np.abs(M)), 1)
        im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("d", DIV),
                       norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    else:
        im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("s", BLUE), vmin=0,
                       vmax=max(np.nanmax(M), 1), aspect="auto")
    for a in range(len(R)):
        for b in range(len(C)):
            v = M[a, b]
            dark = not diverging and not np.isnan(v) and im.norm(v) > 0.6
            ax.text(b, a, "infeas." if np.isnan(v) else f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(C)), [str(c) for c in C])
    ax.set_yticks(range(len(R)), [str(r) for r in R])
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel(cols_key, color=INK2, fontsize=9)
    ax.set_ylabel(rows_key, color=INK2, fontsize=9)
    ax.set_title(f"{metric}  [x100]", color=INK, fontsize=9.5, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#f9f9f7")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--heatmap", nargs=2, metavar=("ROWS", "COLS"))
    ap.add_argument("--metric", default="ES_orC")
    ap.add_argument("--v_max", type=float, default=0.05)
    a = ap.parse_args()
    D = json.load(open(a.json))
    stem = os.path.splitext(a.json)[0]
    if a.heatmap:
        heatmap(D["table"], D["cfgs"], *a.heatmap, a.metric, f"{stem}_{a.metric}.png",
                diverging=a.metric in ("joint", "es_gain"))
    else:
        frontiers(D["table"], a.v_max, f"{stem}_frontier.png", a.configs)
