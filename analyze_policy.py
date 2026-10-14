"""Behaviour of trained policies and of the rule/oracle references on the 30 evaluation
episodes, for the text and figures of the results section:

  - n_on: mean number of active cells over slots and episodes,
  - rel_cio_busy / rel_cio_calm: offset of a hotspot cell minus the mean offset of its
    active neighbours [dB], over slots in which the hotspot cell is congested (load > 1)
    or not.  With incremental CIO actions the absolute offsets drift together, so only
    the offset relative to the neighbours steers users,
  - cio_change: share of active cell-slots in which the offset changes (cell active in both
    slots); the rule adjusts offsets far more often than the learned MLB,
  - trace: active cells per slot in episode 0, with the offered demand of the cluster,
  - fixed plan (--fixed_plan): V of the same activation trajectory replayed with all offsets
    at 0.  evaluate.py --mlb_off instead lets the ES react to the loads it then observes, so
    the ES deactivates more cells and the two effects are mixed.

Learned runs are replayed exactly as evaluate.py does (same episodes, same sampling
generator), so their energy saving and V match eval_stochastic_final.json.

    .venv/bin/python analyze_policy.py --run runs/pidA_ippo_s1 [--run ...]   # -> <run>/behaviour.json
    .venv/bin/python analyze_policy.py --refs [--preset hotspot --base v_max=0.03]
    .venv/bin/python analyze_policy.py --summary "runs/pidA_*_s*" results/behaviour_refs_v0.05.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from baselines import CioOracleMLB, OracleES, RuleES, RuleMLB, ZeroMLB, run
from evaluate import load, policies
from probe import build_configs
from ran_env import EnvConfig, RanEnv
from reference_table import ORACLE, TOP, best_par


def episodes(cfg, make, n_episodes=30, chunk=5, trace_ep=0):
    """make() -> (es, mlb), called once per chunk as in evaluate.py."""
    nbr = RanEnv(cfg, 1, 0).nbr
    n_on, busy, calm, trace, demand = [], [], [], None, None
    changed = []
    for k in range(int(np.ceil(n_episodes / chunk))):
        n = min(chunk, n_episodes - k * chunk)
        es, mlb = make()
        r = run(cfg, es, mlb, n_envs=n, seed=1000 * k, record=True)
        tr = r["trace"]
        s, th, rho = np.asarray(tr["s"]) > 0, np.asarray(tr["theta"]), np.asarray(tr["rho"])   # (T, n, B)
        n_on.extend(s.sum(-1).mean(0).tolist())
        both = s[1:] & s[:-1]
        changed.append((th[1:] != th[:-1])[both])
        for e in range(n):
            for h in r["hot_cells"][e]:
                if h < 0:
                    continue
                act_nb = s[:, e][:, nbr[h]]
                cnt = act_nb.sum(-1)
                mean_nb = np.where(act_nb, th[:, e][:, nbr[h]], 0.0).sum(-1) / np.maximum(cnt, 1)
                rel = th[:, e, h] - mean_nb
                ok = s[:, e, h] & (cnt > 0)
                busy.extend(rel[ok & (rho[:, e, h] > 1.0)].tolist())
                calm.extend(rel[ok & (rho[:, e, h] <= 1.0)].tolist())
        if k * chunk <= trace_ep < k * chunk + n:
            e = trace_ep - k * chunk
            trace = s[:, e].sum(-1).astype(int).tolist()
            env = RanEnv(cfg, n, 1000 * k)
            env.reset()
            demand = (env.demand_at(np.arange(cfg.T))[:, e].sum(-1) / 1e6).tolist()   # Mbps
    return dict(n_on=float(np.mean(n_on)),
                rel_cio_busy=float(np.mean(busy)) if busy else float("nan"),
                rel_cio_calm=float(np.mean(calm)) if calm else float("nan"),
                share_busy=len(busy) / max(len(busy) + len(calm), 1),
                cio_change=float(np.concatenate(changed).mean()),
                trace=trace, demand=demand)


class ReplayES:
    """Requests the recorded activation of each slot; the recorded trajectory already met
    the dwell and coverage rules, so the environment reproduces it."""

    def __init__(self, s_rec):
        self.s_rec = s_rec                                 # (T, n, B)

    def reset(self, env):
        pass

    def observe(self, env, info):
        pass

    def act(self, env):
        return self.s_rec[env.t].astype(int)


def fixed_plan(run_dir, n_episodes=30, chunk=5):
    conf, m, cfg, hp, nets = load(run_dir)
    rng = np.random.default_rng(0)                         # as evaluate.py
    V, V0, same = [], [], []
    for k in range(int(np.ceil(n_episodes / chunk))):
        n = min(chunk, n_episodes - k * chunk)
        es, mlb = policies(m, nets, hp, False, rng)
        r = run(cfg, es, mlb, n_envs=n, seed=1000 * k, record=True)
        s_rec = np.asarray(r["trace"]["s"])
        r0 = run(cfg, ReplayES(s_rec), ZeroMLB(), n_envs=n, seed=1000 * k, record=True)
        same.append(np.array_equal(np.asarray(r0["trace"]["s"]), s_rec))
        V.extend(np.asarray(r["V"]).tolist()), V0.extend(np.asarray(r0["V"]).tolist())
    out = dict(method=conf["method"], seed=conf["seed"], V=float(np.mean(V)),
               V_fixed_plan_theta0=float(np.mean(V0)), plan_reproduced=bool(all(same)),
               ok=int((np.array(V) <= cfg.v_max).sum()), ok_theta0=int((np.array(V0) <= cfg.v_max).sum()))
    json.dump(out, open(os.path.join(run_dir, "fixed_plan.json"), "w"))
    return out


def learned(run_dir):
    conf, m, cfg, hp, nets = load(run_dir)
    rng = np.random.default_rng(0)                         # as evaluate.py (sampled actions)
    out = episodes(cfg, lambda: policies(m, nets, hp, False, rng))
    out.update(method=conf["method"], seed=conf["seed"])
    json.dump(out, open(os.path.join(run_dir, "behaviour.json"), "w"))
    return out


def references(preset, base):
    cfg = EnvConfig(**next(iter(build_configs(preset, base, []).values())))
    rho_th = best_par(TOP[2][1], "ruleT")
    margin = best_par(ORACLE[1][1], "orC")
    if cfg.v_max != 0.05:                                  # pick against this V_max
        probe = {0.03: "results/probe_vmax3.json"}.get(cfg.v_max)
        if probe:
            rho_th, margin = best_par(probe, "ruleT"), best_par(probe, "orC")
    out = {"rule": episodes(cfg, lambda: (RuleES(rho_th), RuleMLB(trigger=0.9))),
           "oracle": episodes(cfg, lambda: (OracleES(margin, "cio", cio_w=0.05),
                                            CioOracleMLB(margin=0.15, every=5, w=0.05, sees="prev")))}
    out["rule"]["par"], out["oracle"]["par"] = rho_th, margin
    path = f"results/behaviour_refs_v{cfg.v_max:g}.json"
    json.dump(out, open(path, "w"))
    return path, out


def summary(patterns):
    rows = {}
    for pat in patterns:
        if pat.endswith(".json") and os.path.exists(pat):
            for name, r in json.load(open(pat)).items():
                rows.setdefault(f"{name} (par {r['par']:g})", []).append(r)
            continue
        for d in sorted(glob.glob(pat)):
            f = os.path.join(d, "behaviour.json")
            if os.path.exists(f):
                prefix = re.match(r"runs/([^_]+)_", d).group(1)
                r = json.load(open(f))
                rows.setdefault(f"{prefix}:{r['method']}", []).append(r)
    print(f"{'':26s} {'n':>2s} {'active cells':>13s} {'rel. CIO busy [dB]':>19s} {'rel. CIO calm [dB]':>19s}"
          f" {'busy share':>10s} {'CIO change':>10s}")
    for lab, rs in rows.items():
        g = lambda k: np.array([r[k] for r in rs], float)
        sd = lambda x: f" ± {x.std(ddof=1):.2f}" if len(x) > 1 else ""
        print(f"{lab:26s} {len(rs):2d} {g('n_on').mean():6.2f}{sd(g('n_on')):7s} "
              f"{np.nanmean(g('rel_cio_busy')):+8.2f}{sd(g('rel_cio_busy')):11s} "
              f"{np.nanmean(g('rel_cio_calm')):+8.2f}{sd(g('rel_cio_calm')):11s} {g('share_busy').mean():10.2f} {100 * np.nanmean(g('cio_change')) if 'cio_change' in rs[0] else float('nan'):9.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--refs", action="store_true", help="rule-based pair and oracle reference")
    ap.add_argument("--fixed_plan", action="append", default=[],
                    help="run dir: V of its activation trajectory replayed with offsets at 0")
    ap.add_argument("--preset", default="hotspot")
    ap.add_argument("--base", default="")
    ap.add_argument("--summary", nargs="*")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    for d in a.run:
        r = learned(d)
        print(f"{d}: active cells {r['n_on']:.2f}, rel. CIO busy {r['rel_cio_busy']:+.2f} dB, "
              f"calm {r['rel_cio_calm']:+.2f} dB")
    for d in a.fixed_plan:
        r = fixed_plan(d)
        print(f"{d}: V {r['V']:.4f}, same plan with θ = 0: {r['V_fixed_plan_theta0']:.4f} "
              f"(plan reproduced: {r['plan_reproduced']})")
    if a.refs:
        path, out = references(a.preset, a.base)
        for k, r in out.items():
            print(f"{k} (par {r['par']}): active cells {r['n_on']:.2f}, rel. CIO busy {r['rel_cio_busy']:+.2f} dB")
        print(path)
    if a.summary:
        summary(a.summary)
