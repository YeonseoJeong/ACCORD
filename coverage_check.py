"""Coverage check: how often does a sleeping cell have no active neighbour?

  1. share of (episode, slot) pairs with at least one uncovered sleeping cell,
     for learned policies, the rule-based pair and the oracle reference
  2. deactivations RuleES gave up because of its coverage guard

    .venv/bin/python coverage_check.py runs/full_accord_s1 runs/decay_accord_s1 ...
"""
import argparse

import numpy as np
import torch

from baselines import CioOracleMLB, OracleES, RuleES, RuleMLB, run
from evaluate import load, policies
from ran_env import RanEnv, make_config


def uncovered(trace_s, nbr):
    """trace_s (T, E, B) bool -> share of slots with an uncovered sleeping cell, and
    share of sleeping cell-slots that are uncovered."""
    on = trace_s.astype(bool)
    has_on_nbr = (on.astype(float) @ nbr.T.astype(float)) > 0      # (T, E, B)
    bad = ~on & ~has_on_nbr
    sleeping = (~on).sum()
    return float(bad.any(-1).mean()), float(bad.sum() / max(sleeping, 1))


def episodes(cfg, make_policies, n=30, chunk=5):
    shares, cells, extra = [], [], []
    nbr = RanEnv(cfg, 1, 0).nbr
    for k in range(n // chunk):
        es, mlb = make_policies()
        r = run(cfg, es, mlb, n_envs=chunk, seed=1000 * k, record=True)
        a, c = uncovered(r["trace"]["s"], nbr)
        shares.append(a), cells.append(c), extra.append(es)
    return np.mean(shares), np.mean(cells), extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--rule", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--preset", default="hotspot")
    a = ap.parse_args()
    torch.set_num_threads(1)
    for rd in a.runs:
        conf, m, cfg, hp, nets = load(rd)
        rng = np.random.default_rng(0)
        sh, ce, _ = episodes(cfg, lambda: policies(m, nets, hp, True, rng))
        print(f"{rd:28s} slots with an uncovered sleeping cell {100 * sh:5.2f}%   "
              f"uncovered share of sleeping cell-slots {100 * ce:5.2f}%", flush=True)
    cfg = make_config(a.preset)
    if a.rule:
        for rho_off, trig in ((0.3, 0.9), (0.4, None), (0.2, None)):
            sh, ce, ess = episodes(cfg, lambda: (RuleES(rho_off), RuleMLB(trigger=trig)))
            nd, ng = sum(e.n_deact for e in ess), sum(e.n_guard for e in ess)
            lab = f"rule ES {rho_off} + MLB rule{' (dead zone)' if trig else ''}"
            print(f"{lab:36s} uncovered {100 * sh:5.2f}%   deactivations {nd:4d}   "
                  f"given up by the coverage guard {ng:4d}", flush=True)
    if a.oracle:
        sh, ce, _ = episodes(cfg, lambda: (OracleES(0.0, "cio"), CioOracleMLB()))
        print(f"{'oracle reference (margin 0)':36s} slots with an uncovered sleeping cell "
              f"{100 * sh:5.2f}%   uncovered share of sleeping cell-slots {100 * ce:5.2f}%",
              flush=True)
