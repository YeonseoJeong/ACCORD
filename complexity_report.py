#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from ran_env import EnvConfig

import numpy as np
import torch


def read_log(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty log: {path}")
    return rows


def farr(rows, key):
    vals = []
    for r in rows:
        s = r.get(key, "")
        vals.append(float(s) if s not in ("", "nan", "NaN", None) else np.nan)
    return np.asarray(vals, dtype=float)


def parse_peak_rss_mb(path):
    if not os.path.exists(path):
        return np.nan
    text = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", text)
    return float(m.group(1)) / 1024.0 if m else np.nan


def count_params_from_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    nets = ckpt["nets"]

    actor_names = {"es_actor", "mlb_actor"}
    critic_names = {"v_e", "v_m", "v_c", "v_cm"}

    def count_group(names):
        total = 0
        for name in names:
            sd = nets.get(name)
            if not isinstance(sd, dict):
                continue
            for value in sd.values():
                if torch.is_tensor(value):
                    total += value.numel()
        return int(total)

    actor = count_group(actor_names)
    critic = count_group(critic_names)
    return actor + critic, actor, critic


def stable_feasible_iter(V, v_max, window=50, hold=20):
    V = np.asarray(V, dtype=float)
    if len(V) < window:
        return np.nan

    ma = np.convolve(V, np.ones(window) / window, mode="valid")
    good = ma <= v_max

    if hold <= 1:
        idx = np.flatnonzero(good)
        return float(idx[0] + window) if len(idx) else np.nan

    if len(good) < hold:
        return np.nan

    streak = np.convolve(good.astype(int), np.ones(hold, dtype=int), mode="valid")
    idx = np.flatnonzero(streak == hold)
    if not len(idx):
        return np.nan

    return float(idx[0] + window)


def one_run(run_dir, warmup=50, window=50, hold=20):
    config_path = os.path.join(run_dir, "config.json")
    log_path = os.path.join(run_dir, "log.csv")
    ckpt_path = os.path.join(run_dir, "final.pt")
    time_path = os.path.join(run_dir, "time.txt")

    conf = json.load(open(config_path))
    rows = read_log(log_path)

    it = farr(rows, "iter")
    sec = farr(rows, "sec")
    V = farr(rows, "V")

    total_sec = float(sec[-1])

    start = min(max(int(warmup), 1), len(sec) - 1)
    denom = it[-1] - it[start - 1]
    sec_per_iter = (
        float((sec[-1] - sec[start - 1]) / denom)
        if denom > 0
        else float(total_sec / max(it[-1], 1))
    )

    peak_rss_mb = parse_peak_rss_mb(time_path)
    total_params, actor_params, critic_params = count_params_from_checkpoint(ckpt_path)

    cfg = EnvConfig(**conf["cfg"])

    v_max = float(cfg.v_max)
    feas_iter = stable_feasible_iter(V, v_max, window=window, hold=hold)

    samples_per_iter = int(cfg.T) * int(conf["hp"]["envs"])
    samples_to_feasible = (
        int(feas_iter * samples_per_iter) if np.isfinite(feas_iter) else np.nan
    )

    return {
        "run": run_dir,
        "method": conf["method"],
        "seed": int(conf["seed"]),
        "iterations": int(it[-1]),
        "total_sec": total_sec,
        "sec_per_iter": sec_per_iter,
        "peak_rss_mb": peak_rss_mb,
        "train_params": total_params,
        "actor_params": actor_params,
        "critic_params": critic_params,
        "stable_feasible_iter": feas_iter,
        "samples_per_iter": samples_per_iter,
        "samples_to_feasible": samples_to_feasible,
    }


def mean_std(xs):
    x = np.asarray(xs, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def fmt(m, s, digits=2):
    if not np.isfinite(m):
        return "N/A"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="complexity")
    ap.add_argument("--methods", nargs="+", default=["ippo", "central"])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--hold", type=int, default=20)
    args = ap.parse_args()

    records = []
    for method in args.methods:
        pat = f"runs/{args.prefix}_{method}_s*"
        dirs = sorted(d for d in glob.glob(pat) if os.path.isdir(d))
        if not dirs:
            print(f"warning: no runs matched {pat}")
        for d in dirs:
            records.append(one_run(d, args.warmup, args.window, args.hold))

    if not records:
        raise SystemExit("No complexity runs found.")

    os.makedirs("results", exist_ok=True)

    per_run_path = f"results/{args.prefix}_complexity_runs.csv"
    with open(per_run_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    groups = defaultdict(list)
    for r in records:
        groups[r["method"]].append(r)

    summary_rows = []
    for method in args.methods:
        rs = groups.get(method, [])
        if not rs:
            continue

        row = {
            "method": method,
            "n_seeds": len(rs),
            "train_params": rs[0]["train_params"],
            "actor_params": rs[0]["actor_params"],
            "critic_params": rs[0]["critic_params"],
        }

        for key in ("total_sec", "sec_per_iter", "peak_rss_mb",
                    "stable_feasible_iter", "samples_to_feasible"):
            m, s = mean_std([r[key] for r in rs])
            row[key + "_mean"] = m
            row[key + "_std"] = s

        summary_rows.append(row)

    summary_path = f"results/{args.prefix}_complexity_summary.csv"
    fields = [
        "method", "n_seeds",
        "train_params", "actor_params", "critic_params",
        "total_sec_mean", "total_sec_std",
        "sec_per_iter_mean", "sec_per_iter_std",
        "peak_rss_mb_mean", "peak_rss_mb_std",
        "stable_feasible_iter_mean", "stable_feasible_iter_std",
        "samples_to_feasible_mean", "samples_to_feasible_std",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    label = {"ippo": "ACCORD", "central": "Centralized"}

    print()
    print("Training complexity summary")
    print("=" * 118)
    print(f"{'Method':<14} {'Train params':>13} {'Actor params':>13} "
          f"{'Time/iter [s]':>20} {'Peak RSS [MB]':>20} {'Stable feas. iter':>20}")
    print("-" * 118)

    for r in summary_rows:
        print(
            f"{label.get(r['method'], r['method']):<14} "
            f"{r['train_params']:>13,d} "
            f"{r['actor_params']:>13,d} "
            f"{fmt(r['sec_per_iter_mean'], r['sec_per_iter_std'], 3):>20} "
            f"{fmt(r['peak_rss_mb_mean'], r['peak_rss_mb_std'], 1):>20} "
            f"{fmt(r['stable_feasible_iter_mean'], r['stable_feasible_iter_std'], 1):>20}"
        )

    print("=" * 118)
    print(f"per-run CSV : {per_run_path}")
    print(f"summary CSV : {summary_path}")
    print(
        f"Stable feasibility = {args.window}-iteration trailing mean(V) <= V_max "
        f"for {args.hold} consecutive rolling windows."
    )


if __name__ == "__main__":
    main()
