"""One-cycle traces of a single episode: active cells, demand profiles, hotspot load.

    python trace_plot.py --preset hotspot --base hotspot_layout=nonadjacent,demand_mbps=2.5
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import CioOracleMLB, OracleES, RuleES, RuleMLB, run
from plot_probe import INK, INK2, style
from probe import build_configs
from ran_env import PRESETS, EnvConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="hotspot", choices=sorted(PRESETS))
    ap.add_argument("--base", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho_off", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--out", default="results/trace.png")
    a = ap.parse_args()
    cfg = EnvConfig(**next(iter(build_configs(a.preset, a.base, []).values())))
    runs = [("Rule pair", "#2a78d6", RuleES(a.rho_off), RuleMLB()),
            ("ES rule + CIO oracle", "#eda100", RuleES(a.rho_off), CioOracleMLB()),
            ("Oracle ES + CIO oracle", "#4a3aa7", OracleES(a.margin, "cio"), CioOracleMLB())]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax in axes:
        style(ax)
    t = np.arange(cfg.T)
    for lab, c, es, mlb in runs:
        r = run(cfg, es, mlb, n_envs=1, seed=a.seed, record=True)
        axes[0].step(t, r["trace"]["n_on"][:, 0], color=c, lw=2, where="post",
                     label=f"{lab}  (Ξ={r['Xi'][0]:.0f}, V̄={r['V'][0]:.3f})")
        rho = r["trace"]["rho"][:, 0]
        hot = r["hot_cells"][0]
        cells = hot[hot >= 0] if (hot >= 0).any() else np.arange(cfg.n_cells)
        axes[2].plot(t, rho[:, cells].max(1), color=c, lw=1, alpha=0.9)
    xi = r["xi"][:, 0]
    for b in range(cfg.n_cells):
        axes[1].plot(t, xi[:, b], color="#898781", lw=1)
    for b in cells:
        axes[1].plot(t, xi[:, b], color=INK, lw=2)
    axes[2].axhline(1 / cfg.beta, color=INK2, ls="--", lw=1)
    axes[2].text(0, 1 / cfg.beta, " 1/β", color=INK2, fontsize=8, va="bottom")
    axes[0].set_ylabel("active cells", color=INK2)
    axes[1].set_ylabel("ξ_b(t)  (hotspot cells bold)", color=INK2)
    axes[2].set_ylabel("max hotspot-cell load", color=INK2)
    axes[2].set_xlabel("slot", color=INK2)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper right")
    axes[0].set_title(f"{a.preset} {a.base}  seed {a.seed}", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor="#f9f9f7")


if __name__ == "__main__":
    main()
