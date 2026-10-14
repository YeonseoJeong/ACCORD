#!/usr/bin/env python3
"""IEEE-style, two-panel population plot with gen_eval baselines.

The dynamic-UE population CSV and Rule-based / All-on baselines can be obtained
from separate evaluation scripts.  Give the fresh gen_eval baseline CSV with
--baseline-csv: it replaces any older Rule-based and All-on rows in --csv.

Examples:
    python plot_population_paper.py \
        --csv run_dynamic_ue_50_100/population_table.csv \
        --baseline-csv run_dynamic_ue_50_100/population_baselines_episodes.csv \
        --out run_dynamic_ue_50_100/population_paper

Baseline-only check (before the dynamic-UE CSV exists):
    python plot_population_paper.py \
        --csv run_dynamic_ue_50_100/population_baselines_episodes.csv \
        --out run_dynamic_ue_50_100/population_baselines_paper

CSV schema: Method, UEs, Avg_Active_BS, Service_Degradation,
            Energy_Saving_pct; optional Episode or *_std columns.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLES = {
    "ACCORD (fixed-100)": dict(label="ACCORD (fixed-100)", color="#7851A9", marker="o", ls="-", lw=1.8),
    "ACCORD (dynamic)": dict(label="ACCORD (dynamic)", color="#0072B2", marker="s", ls="-", lw=2.1),
    "ES-only (dynamic)": dict(label="ES-only (dynamic)", color="#D89000", marker="D", ls="-", lw=1.7),
    "Rule-based": dict(label="Rule-based", color="#008B63", marker="^", ls="-", lw=1.7),
    "All-on": dict(label="All-on", color="#777777", marker=None, ls="--", lw=1.35),
}
METRICS = ("Energy_Saving_pct", "Service_Degradation")
REQUIRED = {"Method", "UEs", "Avg_Active_BS", *METRICS}


def load_data(path: Path):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
    df["Method"] = df["Method"].astype(str).str.strip()
    df["UEs"] = pd.to_numeric(df["UEs"], errors="raise")
    for metric in (*METRICS, "Avg_Active_BS"):
        df[metric] = pd.to_numeric(df[metric], errors="raise")
    df = df.loc[df["Method"].isin(STYLES)].copy()
    if df.empty:
        raise ValueError(f"No methods in {path} match STYLES")

    group_keys = ["Method", "UEs"]
    if "Episode" in df.columns:
        # A single method/N has one observation per evaluation episode; no
        # artificial std if only an aggregated mean is available.
        if df.duplicated(group_keys + ["Episode"]).any():
            raise ValueError(f"{path}: repeated Method/UEs/Episode rows")
        summary = df.groupby(group_keys, as_index=False).agg(
            **{m: (m, "mean") for m in METRICS},
            **{m + "_std": (m, "std") for m in METRICS},
        )
    else:
        if df.duplicated(group_keys).any():
            raise ValueError(f"{path}: repeated Method/UEs rows without Episode column")
        summary = df[group_keys + list(METRICS) + [c for c in df if c.endswith("_std")]].copy()
        for m in METRICS:
            if m + "_std" not in summary:
                summary[m + "_std"] = np.nan
            else:
                summary[m + "_std"] = pd.to_numeric(summary[m + "_std"], errors="raise")
    return summary


def make_figure(summary, out: Path, vmax=0.05, width=5.6):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.4,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.75), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.975, top=0.815, bottom=0.125, hspace=0.19)
    xs = np.sort(summary["UEs"].unique())
    legend_handles, legend_labels = [], []
    has_any_std = False

    for method, style in STYLES.items():
        sub = summary.loc[summary["Method"] == method].sort_values("UEs")
        if sub.empty:
            continue
        x = sub["UEs"].to_numpy(float)
        for axis, metric in zip(axes, METRICS):
            y = sub[metric].to_numpy(float)
            (line,) = axis.plot(
                x, y, label=style["label"], color=style["color"],
                linestyle=style["ls"], linewidth=style["lw"],
                marker=style["marker"], markersize=4.2,
                markeredgecolor="white" if style["marker"] else style["color"],
                markeredgewidth=0.45,
                zorder=4 if method.startswith("ACCORD") else 3,
            )
            sd = sub[metric + "_std"].to_numpy(float)
            valid = np.isfinite(sd)
            if valid.any():
                has_any_std = True
                lo = np.maximum(0, y - sd) if metric == "Service_Degradation" else y - sd
                axis.fill_between(x, lo, y + sd, where=valid,
                                  color=style["color"], alpha=0.10, linewidth=0)
            if axis is axes[0]:
                legend_handles.append(line)
                legend_labels.append(style["label"])

    axes[0].axhline(0, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
    axes[1].axhline(vmax, color="#B2182B", linewidth=1.15, linestyle=":", zorder=2)
    axes[1].annotate(
        rf"$V_{{\max}}={vmax:.2f}$", xy=(0.98, vmax),
        xycoords=("axes fraction", "data"), xytext=(0, 5),
        textcoords="offset points", ha="right", va="bottom",
        color="#B2182B", fontsize=7.5,
    )
    axes[0].set_ylabel("Energy saving (%)")
    axes[1].set_ylabel(r"Service degradation $\bar V$")
    axes[1].set_xlabel("Number of active UEs")
    axes[0].set_ylim(-8, 51)
    axes[0].set_yticks([0, 10, 20, 30, 40, 50])
    max_v = float(summary["Service_Degradation"].max())
    v_upper = max(0.12, np.ceil((max_v + 0.025) * 10) / 10)
    axes[1].set_ylim(0, v_upper)
    axes[1].set_yticks(np.arange(0, v_upper + 1e-9, 0.1 if v_upper > 0.25 else 0.025))
    axes[1].set_xticks(xs)
    axes[1].set_xlim(xs.min() - 5, xs.max() + 5)

    for i, axis in enumerate(axes):
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.65)
        axis.tick_params(length=3.5, width=0.8, direction="out")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(0.01, 0.96, f"({chr(97 + i)})", transform=axis.transAxes,
                  va="top", ha="left", fontsize=9, fontweight="bold")

    if len(legend_handles) == 5:
        order = [0, 3, 1, 4, 2]
        legend_handles = [legend_handles[i] for i in order]
        legend_labels = [legend_labels[i] for i in order]
    fig.legend(legend_handles, legend_labels, loc="upper center",
               bbox_to_anchor=(0.55, 0.985), ncol=3,
               frameon=False, handlelength=2.5, columnspacing=1.4,
               handletextpad=0.5, borderaxespad=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("Saved:", ", ".join(str(out.with_suffix(x)) for x in (".pdf", ".png", ".svg")))
    if not has_any_std:
        print("Note: mean-only CSV input; no uncertainty bands were drawn.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path, help="Dynamic UE population CSV")
    parser.add_argument("--baseline-csv", type=Path, default=None,
                        help="Fresh gen_eval Rule-based/All-on CSV (raw episodes or means)")
    parser.add_argument("--out", type=Path, default=Path("run_dynamic_ue_50_100/population_paper"))
    parser.add_argument("--vmax", type=float, default=0.05)
    parser.add_argument("--width", type=float, default=5.6)
    args = parser.parse_args()
    if args.vmax < 0 or args.width <= 0:
        parser.error("--vmax must be nonnegative and --width must be positive")

    summary = load_data(args.csv)
    if args.baseline_csv is not None:
        baseline = load_data(args.baseline_csv)
        baseline = baseline[baseline["Method"].isin(["Rule-based", "All-on"])].copy()
        if baseline.empty:
            raise ValueError("--baseline-csv contains no Rule-based or All-on rows")
        summary = pd.concat([
            summary[~summary["Method"].isin(["Rule-based", "All-on"])], baseline
        ], ignore_index=True)
        if summary.duplicated(["Method", "UEs"]).any():
            raise ValueError("Duplicate Method/UEs after replacing baseline data")
        print("Rule-based and All-on rows replaced with gen_eval data:", args.baseline_csv)
    make_figure(summary, args.out, args.vmax, args.width)


if __name__ == "__main__":
    main()
