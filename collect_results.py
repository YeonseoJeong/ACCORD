"""Collect evaluations of trained runs into the results table (mean ± std over seeds).

Reads runs/<prefix>_<method>_s<seed>/eval_<tag>_final.json (tag 'stochastic' by default, the
sampled execution of Algorithm 2; or 'greedy') and eval_<tag>_mlboff_final.json, adds the
non-learning rows of reference_table.py, and prints Markdown and LaTeX rows.

    .venv/bin/python collect_results.py                 # runs/main_*
    .venv/bin/python collect_results.py --prefix dd22
    .venv/bin/python collect_results.py --probe results/probe_hotspot_d2.2.json
    .venv/bin/python collect_results.py --override ippo=mainmd team=mainmd central=mainmd

Prints two tables: Main table (baselines, ACCORD = DTDE, Centralized as reference) and Ablation
(ACCORD with one design element changed: CTDE, team reward, symmetric lambda, w_h = 0, and
extra rows such as the PID multiplier via --extra 'ACCORD (PID λ)=pidA:ippo').
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from reference_table import collect, sources

# ACCORD is the DTDE structure (method key ippo: actors and critics on each agent's own
# observation).  Main table: the baselines (other approaches: one function learns, the other is
# a rule or absent), ACCORD, and Centralized as the non-deployable reference.  Ablation: ACCORD
# with one design element changed (accord = the CTDE structure with global observations and
# critics).  The older CTDE-based variants (team, accord_sym, ...) keep their own labels.
MAIN = ["es_only_i", "es_cio0_i", "ippo", "central"]
ABLATION = ["ippo", "accord", "dtde_team", "dtde_sym", "dtde_wh0"]
LABELS = {"es_only_i": "ES-only learning", "es_cio0_i": "Learned ES, MLB off", "ippo": "ACCORD",
          "es_only": "ES-only learning (PID λ)", "es_cio0": "Learned ES, MLB off (PID λ)",
          "central": "Centralized", "accord": "ACCORD (CTDE)",
          "dtde_team": "ACCORD (team reward)", "dtde_sym": "ACCORD (symmetric λ)",
          "dtde_wh0": "ACCORD, w_h = 0",
          "team": "ACCORD (CTDE, team reward)", "accord_sym": "ACCORD (CTDE, symmetric λ)",
          "accord_no_theta_es": "ACCORD (CTDE) w/o θ at ES",
          "accord_no_tau_mlb": "ACCORD (CTDE) w/o τ at MLB", "accord_wh0": "ACCORD (CTDE), w_h = 0"}
MAIN_LABELS = LABELS
KEYS = ["ES", "V", "ok", "O", "H", "Xi", "short", "jain"]


def gather(prefix, tag="greedy"):
    out = {}
    for path in glob.glob(f"runs/{prefix}_*_s*/eval_{tag}_final.json"):
        m = re.match(rf"runs/{prefix}_(.+)_s(\d+)/", path)
        if m:
            out.setdefault(m.group(1), []).append(json.load(open(path)))
    return out


def stats(rs, k):
    x = np.array([r[k] for r in rs], float)
    if k == "short":
        x = 100 * x
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else 0.0)


def fmt(rs, k, digits):
    m, s = stats(rs, k)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if len(rs) > 1 else f"{m:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="main")
    ap.add_argument("--override", nargs="*", default=[],
                    help="method=prefix for methods taken from another batch, e.g. accord=mainmd")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra ablation rows 'label=prefix:method', e.g. 'ACCORD (PID λ)=pidA:ippo'")
    ap.add_argument("--probe", default=None, help="probe.py output for the non-learning rows")
    ap.add_argument("--v_max", type=float, default=0.05)
    ap.add_argument("--tag", default="stochastic", choices=["stochastic", "greedy"],
                    help="execution the table reports (stochastic = sampled actions, Algorithm 2)")
    a = ap.parse_args()
    ov = dict(x.split("=") for x in a.override)
    extra = [(x.split("=")[0], *x.split("=")[1].split(":")) for x in a.extra]   # (label, prefix, method)
    other = "greedy" if a.tag == "stochastic" else "stochastic"
    data = {p: {t: gather(p, t) for t in (a.tag, other, f"{a.tag}_mlboff")}
            for p in {a.prefix, *ov.values(), *(e[1] for e in extra)}}
    top, ora = sources(a.probe)
    head = ["Method", "runs", "seeds", "Energy saving [%]", "V̄", "seeds with V̄ ≤ V_max",
            "Feasible episodes", "H̄", "Ξ_T", "Short [%]", "Jain", f"V̄ {other}",
            f"{a.tag} − {other} V̄", "V̄ with the MLB off"]

    def probe_row(lab, f, fam, par):
        mm = collect(f, fam, par)
        short = "--" if mm["Xi"] == 0 else f"{100 * mm['short']:.0f}"
        print(f"| {lab} | probe | -- | {mm['ES']:.1f} | {mm['V']:.3f} | -- | {mm['ok']}/{mm['n']} | "
              f"{mm['H']:.3f} | {mm['Xi']:.1f} | {short} | {mm['jain']:.3f} | -- | -- | -- |")

    def learned_rows(methods, tex, labels=LABELS):
        for item in methods:
            m, p = (item, ov.get(item, a.prefix)) if isinstance(item, str) else item[2:]
            if not isinstance(item, str):
                labels = dict(labels, **{m: item[0]})
            rs = data[p][a.tag].get(m)
            if not rs:
                continue
            rs = sorted(rs, key=lambda r: r["seed"])
            st = {r["seed"]: r for r in data[p][other].get(m, [])}
            off = data[p][f"{a.tag}_mlboff"].get(m)
            n_feas = sum(r["V"] <= a.v_max for r in rs)
            n_ok = np.mean([r["ok"] for r in rs])
            pairs = [(r["V"], st[r["seed"]]["V"]) for r in rs if r["seed"] in st]
            v_st = f"{np.mean([q for _, q in pairs]):.3f}" if pairs else "--"
            gap = f"{np.mean([g - q for g, q in pairs]):+.3f}" if pairs else "--"
            off_v = fmt(off, "V", 3) if off else "--"
            print(f"| {labels.get(m, m)} | {p} | {len(rs)} | {fmt(rs, 'ES', 1)} | {fmt(rs, 'V', 3)} | "
                  f"{n_feas}/{len(rs)} | {n_ok:.1f}/30 | {fmt(rs, 'H', 3)} | {fmt(rs, 'Xi', 1)} | "
                  f"{fmt(rs, 'short', 0)} | {fmt(rs, 'jain', 3)} | {v_st} | {gap} | {off_v} |")
            tex.append(f"{labels.get(m, m)} & {stats(rs, 'ES')[0]:.1f} & {stats(rs, 'V')[0]:.3f} & "
                       f"{n_ok:.1f}/30 & {stats(rs, 'O')[0]:.3f} & {stats(rs, 'H')[0]:.3f} & "
                       f"{stats(rs, 'Xi')[0]:.1f} & {stats(rs, 'short')[0]:.0f}\\% & "
                       f"{stats(rs, 'jain')[0]:.3f} & {off_v.split(' ')[0]} \\\\")

    def table(title, rows):
        print(f"\n### {title}\n")
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        tex = []
        rows(tex)
        print(f"\nLaTeX rows ({title.lower()}, means over seeds):")
        print("\n".join(tex))

    def main_rows(tex):
        for r in top:
            probe_row(*r)
        learned_rows(MAIN, tex, MAIN_LABELS)
        for lab, *r in ora:
            probe_row(f"{lab} (reference)", *r)

    methods = set(data[a.prefix][a.tag]) | set(ov)
    known = set(MAIN) | set(ABLATION) | {"es_only", "es_cio0"}
    table("Main table", main_rows)
    table("Ablation", lambda tex: learned_rows(
        ABLATION + [(lab, None, m, p) for lab, p, m in extra] + sorted(methods - known), tex))


if __name__ == "__main__":
    main()
