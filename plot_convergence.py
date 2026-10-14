"""Training convergence: multiplier and smoothed violation per iteration, seed mean (line)
and seed range (band), one series per method (from runs/<run>/log.csv).

    .venv/bin/python plot_convergence.py --out results/convergence_main.png \
        "ACCORD=runs/pidA_ippo_s*" "Centralized=runs/pidA_central_s*" \
        "Learned ES, MLB off=runs/pidA_es_cio0_i_s*" "ES-only learning=runs/pidA_es_only_i_s*"
"""
import argparse
import csv
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_probe import INK, INK2, MUTED, style

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]    # fixed categorical order


def series(pattern, key, w=10):
    runs = [np.array([float(r[key]) for r in csv.DictReader(open(d + "/log.csv"))])
            for d in sorted(glob.glob(pattern)) if not d.endswith((".log", ".txt"))]
    n = min(len(x) for x in runs)
    X = np.stack([np.convolve(x[:n], np.ones(w) / w, mode="valid") for x in runs])
    return np.arange(X.shape[1]) + w, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="label=glob of run directories")
    ap.add_argument("--v_max", type=float, default=0.05)
    ap.add_argument("--out", default="results/convergence_main.png")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax in axes:
        style(ax)
    for i, g in enumerate(a.groups):
        lab, pat = g.split("=", 1)
        c = COLORS[i % len(COLORS)]
        for ax, key in zip(axes, ("lam", "C_hat")):
            x, X = series(pat, key)
            ax.fill_between(x, X.min(0), X.max(0), color=c, alpha=0.18, lw=0)
            ax.plot(x, X.mean(0), color=c, lw=1.8, label=f"{lab} ({len(X)} seeds)")
    axes[0].set_title("multiplier λ", color=INK, fontsize=10, loc="left")
    axes[1].set_title("smoothed violation Ĉ", color=INK, fontsize=10, loc="left")
    axes[1].axhline(a.v_max, color=INK2, ls="--", lw=1)
    axes[1].text(5, a.v_max, " V_max", color=INK2, fontsize=8, va="bottom")
    axes[1].set_ylim(0, 0.15)
    for ax in axes:
        ax.set_xlabel("iteration", color=INK2, fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(a.out)


if __name__ == "__main__":
    main()
