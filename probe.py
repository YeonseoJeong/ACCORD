"""Probe environment settings with non-learning controllers, to find where a
learned, coordinated ES/MLB pair (ACCORD) has room to beat simple baselines.

Families (each traced over its parameter, best point with mean V <= V_max kept)

  allon       All-on, CIO = 0                           energy reference
  allon_rule  All-on + RuleMLB
  allon_ruleT All-on + RuleMLB with a 0.9 dead zone
  allon_cio   All-on + CioOracleMLB                     what the CIO alone can do
  rule        RuleES + RuleMLB   (rho_off sweep)        the paper's rule-based pair
  ruleT       RuleES + RuleMLB with a 0.9 dead zone     a stronger rule-based pair
  ruleES0     RuleES, CIO = 0    (rho_off sweep)
  ruleC       RuleES + CioOracleMLB (rho_off sweep)     simple ES on top of an ideal MLB
  or0         OracleES + CIO = 0        (margin sweep)
  orR         OracleES + RuleMLB        (margin sweep)  ~ "ES-only learning"
  orC         OracleES + CioOracleMLB   (margin sweep)  ~ joint / centralized bound

  headroom = ES*(orC) - max(ES* of rule, ruleT, ruleES0) room over rule-based
  es_gain  = ES*(orC) - ES*(ruleC)                      what a smarter ES adds
  joint    = ES*(orC) - ES*(orR)                        ACCORD vs ES-only learning

Seeds are split into chunks that run in parallel; every family sees the same
episodes.  Episode-to-episode spread is large, so use >= 30 seeds.  Keys
cio_m, cio_every, cio_w and cio_sees in --base/--sweep set the CIO oracle
(margin, re-optimisation period, excess weight, 'prev' = sees s_{t-1} as in the
paper's simultaneous moves or 'next' = sees s_t) instead of the environment.

    python probe.py --preset hotspot --sweep demand_mbps=2.2,2.4,2.6 --fams allon_rule,allon_cio
    python probe.py --preset hotspot --base demand_mbps=2.5 --seeds 30
"""
import argparse
import itertools
import json
import os
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np

from baselines import (AllOnES, CioOracleMLB, OracleES, RuleES, RuleMLB, ZeroMLB, run)
from ran_env import PRESETS, EnvConfig

RHO_OFF = [0.1, 0.2, 0.3, 0.4, 0.5]
MARGINS = [0.0, 0.1, 0.2]
def _cio(ck):
    return CioOracleMLB(margin=ck.get("m", 0.15), every=ck.get("every", 5), w=ck.get("w", 0.05),
                        sees=ck.get("sees", "prev"))


# p: family parameter; ck: settings of the CIO oracle (keys m, every, w, sees)
FAMILIES = {
    "allon": lambda p, ck: (AllOnES(), ZeroMLB()),
    "allon_rule": lambda p, ck: (AllOnES(), RuleMLB()),
    "allon_ruleT": lambda p, ck: (AllOnES(), RuleMLB(trigger=0.9)),
    "allon_cio": lambda p, ck: (AllOnES(), _cio(ck)),
    "rule": lambda p, ck: (RuleES(p), RuleMLB()),
    "ruleT": lambda p, ck: (RuleES(p), RuleMLB(trigger=0.9)),
    "ruleES0": lambda p, ck: (RuleES(p), ZeroMLB()),
    "ruleC": lambda p, ck: (RuleES(p), _cio(ck)),
    "or0": lambda p, ck: (OracleES(p, "zero"), ZeroMLB()),
    "orR": lambda p, ck: (OracleES(p, "rule"), RuleMLB()),
    "orC": lambda p, ck: (OracleES(p, "cio", cio_w=ck.get("w", 0.05)), _cio(ck)),
}


def split_kw(kw):
    """Config keys starting with 'cio_' set the CIO oracle, the rest the env."""
    ck = {k[4:]: v for k, v in kw.items() if k.startswith("cio_")}
    return {k: v for k, v in kw.items() if not k.startswith("cio_")}, ck
PARAMS = {"rule": RHO_OFF, "ruleT": RHO_OFF, "ruleES0": RHO_OFF, "ruleC": RHO_OFF,
          "or0": MARGINS, "orR": MARGINS, "orC": MARGINS}
SLOW = {"or0": 3, "orR": 3, "orC": 4, "ruleC": 2, "allon_cio": 2}   # for job ordering


def _job(args):
    cname, cfg_kw, fam, par, chunk, n_envs, seed = args
    env_kw, ck = split_kw(cfg_kw)
    es, mlb = FAMILIES[fam](par, ck)
    t0 = time.time()
    r = run(EnvConfig(**env_kw), es, mlb, n_envs=n_envs, seed=seed + 1000 * chunk)
    keep = ["E_sum", "V", "H", "O", "Xi", "jain", "n_off", "n_short", "hot_cells"]
    out = {k: np.asarray(r[k]).tolist() for k in keep}
    out.update(config=cname, fam=fam, par=par, chunk=chunk, sec=time.time() - t0)
    return out


def merge(rows):
    """Concatenate chunks in chunk order -> one record per (config, fam, par)."""
    groups = {}
    for r in sorted(rows, key=lambda r: r["chunk"]):
        g = groups.setdefault((r["config"], r["fam"], r["par"]), {})
        for k, v in r.items():
            if isinstance(v, list):
                g.setdefault(k, []).extend(v)
    return groups


def summarize(groups, cfgs):
    table = []
    for cname in cfgs:
        v_max = EnvConfig(**split_kw(cfgs[cname])[0]).v_max
        e0 = np.array(groups[(cname, "allon", None)]["E_sum"])
        pts = []
        for (c, fam, par), g in groups.items():
            if c != cname:
                continue
            V = np.array(g["V"])
            pts.append(dict(fam=fam, par=par, ES=float(np.mean(1 - np.array(g["E_sum"]) / e0)),
                            V=float(V.mean()), V_std=float(V.std()),
                            n_feas=int((V <= v_max).sum()), n=len(V),
                            H=float(np.mean(g["H"])), O=float(np.mean(g["O"])),
                            Xi=float(np.mean(g["Xi"])), n_off=float(np.mean(g["n_off"])),
                            n_short=float(np.mean(g["n_short"]))))
        allon = {p["fam"]: p for p in pts if p["fam"].startswith("allon")}
        row = dict(config=cname, points=pts)
        for k, p in allon.items():
            row[f"V_{k}"], row[f"feas_{k}"] = p["V"], f"{p['n_feas']}/{p['n']}"
        for fam in PARAMS:
            cand = [p for p in pts if p["fam"] == fam]
            if not cand:
                continue
            ok = [p for p in cand if p["V"] <= v_max]
            if ok:
                b = max(ok, key=lambda p: p["ES"])
                row[f"ES_{fam}"], row[f"par_{fam}"] = b["ES"], b["par"]
                row[f"V_{fam}"], row[f"feas_{fam}"] = b["V"], f"{b['n_feas']}/{b['n']}"
            else:   # no feasible point: fall back to All-on if that is feasible
                row[f"ES_{fam}"] = 0.0 if allon["allon"]["V"] <= v_max else np.nan
        rule_es = [row[f"ES_{f}"] for f in ("rule", "ruleT", "ruleES0") if f"ES_{f}" in row]
        rule_best = np.nanmax(rule_es) if rule_es and np.isfinite(rule_es).any() else np.nan
        if "ES_orC" in row:
            row["headroom"] = row["ES_orC"] - rule_best
            row["es_gain"] = row["ES_orC"] - row.get("ES_ruleC", np.nan)
            row["joint"] = row["ES_orC"] - row.get("ES_orR", np.nan)
        table.append(row)
    return table


def print_table(table):
    cols = ["V_allon", "V_allon_rule", "V_allon_ruleT", "V_allon_cio", "ES_rule", "ES_ruleT",
            "ES_ruleES0", "ES_ruleC",
            "ES_or0", "ES_orR", "ES_orC", "headroom", "es_gain", "joint"]
    cols = [c for c in cols if any(c in r for r in table)]
    w = max(len(r["config"]) for r in table) + 2
    print(f"{'config':<{w}s}" + "".join(f"{c:>14s}" for c in cols))
    for r in table:
        cells = []
        for c in cols:
            v = r.get(c, np.nan)
            cells.append(f"{v:14.3f}" if v == v else f"{'infeas.':>14s}")
        print(f"{r['config']:<{w}s}" + "".join(cells))
    print("\nper family: best feasible parameter, its mean V and feasible episodes")
    for r in table:
        print(f"  {r['config']}")
        for k in sorted(r):
            if k.startswith("feas_"):
                fam = k[5:]
                par = r.get(f"par_{fam}")
                print(f"    {fam:10s} par={par!s:6s} V={r.get(f'V_{fam}', np.nan):.4f} "
                      f"episodes ok={r[k]}")


def _val(x):
    for f in (int, float):
        try:
            return f(x)
        except ValueError:
            pass
    return x


def build_configs(preset, base, sweeps):
    kw0 = dict(PRESETS[preset])
    if base:
        kw0.update({k: _val(v) for k, v in (kv.split("=") for kv in base.split(","))})
    axes = [(s.split("=")[0], [_val(v) for v in s.split("=")[1].split(",")]) for s in sweeps]
    out = {}
    for combo in itertools.product(*[v for _, v in axes]):
        kw = dict(kw0, **{k: v for (k, _), v in zip(axes, combo)})
        name = "_".join(f"{k}{v}" for (k, _), v in zip(axes, combo)) or preset
        out[name] = kw
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="paper", choices=sorted(PRESETS))
    ap.add_argument("--base", default="", help="overrides on the preset: k=v,k=v")
    ap.add_argument("--sweep", nargs="*", default=[], help="k=v1,v2 ...")
    ap.add_argument("--fams", default="all", help="comma list of families, or 'all'")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=5, help="episodes per job")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--name", default=None)
    ap.add_argument("--margins", default=None, help="override MARGINS, e.g. -0.2,-0.1")
    ap.add_argument("--rho_offs", default=None, help="override RHO_OFF, e.g. 0.2,0.3")
    a = ap.parse_args()
    for key, arg in (("margins", a.margins), ("rho_offs", a.rho_offs)):
        if arg:
            vals = [float(x) for x in arg.split(",")]
            fams_of = ("or0", "orR", "orC") if key == "margins" else ("rule", "ruleT", "ruleES0", "ruleC")
            for f in fams_of:
                PARAMS[f] = vals

    cfgs = build_configs(a.preset, a.base, a.sweep)
    fams = list(FAMILIES) if a.fams == "all" else ["allon"] + [
        f for f in a.fams.split(",") if f != "allon"]
    n_chunks = int(np.ceil(a.seeds / a.chunk))
    jobs = [(c, kw, f, p, ch, min(a.chunk, a.seeds - ch * a.chunk), a.seed)
            for c, kw in cfgs.items() for f in fams for p in PARAMS.get(f, [None])
            for ch in range(n_chunks)]
    jobs.sort(key=lambda j: -SLOW.get(j[2], 1))
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    table = summarize(merge(rows), cfgs)
    name = a.name or a.preset + ("_" + "_".join(cfgs) if a.sweep and len(cfgs) < 4 else "")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(dict(cfgs=cfgs, rows=rows, table=table), f)
    print_table(table)
    print(f"\n{len(jobs)} jobs, {time.time() - t0:.0f}s  ->  results/{name}.json")
