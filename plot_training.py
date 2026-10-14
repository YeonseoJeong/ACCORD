"""Training curves from runs/<name>/log.csv: lambda, C_hat vs V_max, active cells,
share of zero CIO on active cells.

    .venv/bin/python plot_training.py runs/full_accord_s1 runs/full_es_only_s1 --out results/train.png
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_probe import INK, INK2, style

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]


def load(run):
    rows = list(csv.DictReader(open(os.path.join(run, "log.csv"))))
    out = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
           for k in rows[0]}
    out["method"] = json.load(open(os.path.join(run, "config.json")))["method"]
    return out


def smooth(x, w=10):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/train.png")
    ap.add_argument("--v_max", type=float, default=0.05)
    a = ap.parse_args()
    panels = [("lam", "λ"), ("C_hat", "Ĉ (smoothed V)"), ("n_on", "active cells"),
              ("th_zero", "share of θ = 0 on active cells")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax in axes:
        style(ax)
    for i, run in enumerate(a.runs):
        d = load(run)
        c = COLORS[i % len(COLORS)]
        for ax, (k, lab) in zip(axes, panels):
            y = d[k]
            if np.all(np.isnan(y)):
                continue
            x = d["iter"][len(d["iter"]) - len(smooth(y)):]
            ax.plot(x, smooth(y), color=c, lw=2, label=d["method"])
            ax.set_title(lab, color=INK, fontsize=10, loc="left")
            ax.set_xlabel("iteration", color=INK2, fontsize=9)
    axes[1].axhline(a.v_max, color=INK2, ls="--", lw=1)
    axes[1].text(0, a.v_max, " V_max", color=INK2, fontsize=8, va="bottom")
    axes[-1].legend(fontsize=8, frameon=False, labelcolor=INK2, loc="best")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor="#f9f9f7")
    print(a.out)


if __name__ == "__main__":
    main()
