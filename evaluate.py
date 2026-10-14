"""Evaluate trained policies on the fixed evaluation episodes.

Episodes are the same as for the non-learning baselines: chunk k = 0..5 is
RanEnv(cfg, n_envs=5, seed=1000 * k).  Metrics match reference_table.py.

    .venv/bin/python evaluate.py runs/accord_s0 [runs/accord_s1 ...] [--stochastic]
"""
import argparse
import json
import os

import numpy as np
import torch

from baselines import AllOnES, RuleMLB, ZeroMLB, run
from ran_env import EnvConfig
from trainer import (HP, METHODS, Nets, ObsSpec, obs_parts, sample_es, sample_mlb,
                     theta_from)
from ran_env import RanEnv

class LearnedES:
    def __init__(self, nets, spec, greedy, rng):
        self.nets, self.spec, self.greedy, self.rng = nets, spec, greedy, rng

    def reset(self, env):
        pass

    def observe(self, env, info):
        pass

    def act(self, env):
        o = self.spec.es(obs_parts(env))
        a, _ = sample_es(self.nets, o, env.eligible(), env.s, self.rng, self.greedy)
        return a.astype(int)


class LearnedMLB:
    def __init__(self, nets, spec, hp, greedy, rng):
        self.nets, self.spec, self.hp, self.greedy, self.rng = nets, spec, hp, greedy, rng

    def reset(self, env):
        pass

    def observe(self, env, info):
        pass

    def act(self, env, s_next):
        o = self.spec.mlb(obs_parts(env), s_next)
        idx, _ = sample_mlb(self.nets, o, self.rng, self.greedy)
        return theta_from(self.nets, idx, env.theta, env.cfg, self.hp.mlb_action)


def load(run_dir, ckpt="final.pt"):
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    m = METHODS[conf["method"]]
    cfg = EnvConfig(**conf["cfg"])
    hp = HP(**conf["hp"])
    env = RanEnv(cfg, 1, 0)
    env.reset()
    nets = Nets(m, cfg, hp, env)
    sd = torch.load(os.path.join(run_dir, ckpt), weights_only=False)
    nets.load_state_dict(sd["nets"])
    return conf, m, cfg, hp, nets


def policies(m, nets, hp, greedy, rng, mlb_off=False):
    spec = ObsSpec(m, hp, nets.nbr)
    es = LearnedES(nets, spec, greedy, rng) if m["es"] == "learn" else AllOnES()
    if mlb_off:
        mlb = ZeroMLB()
    elif m["mlb"] == "learn":
        mlb = LearnedMLB(nets, spec, hp, greedy, rng)
    elif m["mlb"] == "rule_dz":
        mlb = RuleMLB(trigger=0.9)
    else:
        mlb = ZeroMLB()
    return es, mlb


def evaluate(run_dir, episodes=30, chunk=5, greedy=True, ckpt="final.pt", mlb_off=False):
    conf, m, cfg, hp, nets = load(run_dir, ckpt)
    rng = np.random.default_rng(0)
    keys = ("E_sum", "V", "H", "O", "Xi", "jain", "n_off", "n_short")
    acc = {k: [] for k in keys}
    e0 = []
    th_all, hot_rows = [], []
    for k in range(int(np.ceil(episodes / chunk))):
        n = min(chunk, episodes - k * chunk)
        es, mlb = policies(m, nets, hp, greedy, rng, mlb_off)
        r = run(cfg, es, mlb, n_envs=n, seed=1000 * k, record=True)
        for key in keys:
            acc[key].extend(np.asarray(r[key]).tolist())
        tr = r["trace"]                                          # (T, n, B)
        th_all.append(np.where(tr["s"] > 0, tr["theta"], np.nan).reshape(-1, cfg.n_cells))
        hot = np.zeros((n, cfg.n_cells), bool)
        for e in range(n):
            hot[e, r["hot_cells"][e][r["hot_cells"][e] >= 0]] = True
        hot_rows.append((tr["theta"], tr["rho"], tr["s"] > 0, np.broadcast_to(hot, tr["s"].shape)))
        e0.extend(run(cfg, AllOnES(), ZeroMLB(), n_envs=n, seed=1000 * k)["E_sum"].tolist())
    g = {k: np.array(v) for k, v in acc.items()}
    res = dict(method=conf["method"] + ("+mlb_off" if mlb_off else ""), seed=conf["seed"],
               ckpt=ckpt, greedy=greedy,
               ES=float(100 * np.mean(1 - g["E_sum"] / np.array(e0))), V=float(g["V"].mean()),
               ok=int((g["V"] <= cfg.v_max).sum()), n=len(g["V"]), O=float(g["O"].mean()),
               H=float(g["H"].mean()), Xi=float(g["Xi"].mean()), jain=float(g["jain"].mean()),
               short=float(g["n_short"].sum() / max(g["n_off"].sum(), 1)),
               V_per_episode=g["V"].tolist(), theta=theta_stats(th_all, hot_rows, cfg))
    tag = ("greedy" if greedy else "stochastic") + ("_mlboff" if mlb_off else "")
    json.dump(res, open(os.path.join(run_dir, f"eval_{tag}_{ckpt.replace('.pt', '')}.json"), "w"))
    return res


def theta_stats(th_all, hot_rows, cfg):
    """CIO applied on active cells: zero share, histogram, and the mean offset of
    hotspot / other cells split by whether the cell is loaded above 1.0."""
    th = np.concatenate(th_all).ravel()
    th = th[np.isfinite(th)]
    grid = np.arange(cfg.cio_min, cfg.cio_max + 1e-9, cfg.cio_step)
    hist = {f"{g:+.0f}": float(np.mean(np.isclose(th, g))) for g in grid}
    out = dict(zero_share=float(np.mean(th == 0)), mean_abs=float(np.mean(np.abs(th))), hist=hist)
    theta = np.concatenate([h[0].reshape(-1) for h in hot_rows])
    rho = np.concatenate([h[1].reshape(-1) for h in hot_rows])
    act = np.concatenate([h[2].reshape(-1) for h in hot_rows])
    hot = np.concatenate([np.ascontiguousarray(h[3]).reshape(-1) for h in hot_rows])
    for name, sel in (("hot", hot), ("other", ~hot)):
        for lab, cond in (("busy", rho > 1.0), ("calm", rho <= 1.0)):
            mask = act & sel & cond
            out[f"mean_{name}_{lab}"] = float(theta[mask].mean()) if mask.any() else float("nan")
            out[f"share_{name}_{lab}"] = float(mask.mean())
    # when does the MLB intervene: slots where some active cell is congested vs not,
    # and the offsets of the other cells while a hotspot is congested
    cong, hbusy = [], []
    for th_, rho_, act_, hot_ in hot_rows:
        load = rho_ * act_
        c = np.broadcast_to((load.max(-1) > 1.0)[..., None], rho_.shape)
        h = np.broadcast_to(((load * hot_).max(-1) > 1.0)[..., None], rho_.shape)
        cong.append(c.reshape(-1)), hbusy.append(h.reshape(-1))
    cong, hbusy = np.concatenate(cong), np.concatenate(hbusy)
    nz = theta != 0
    for lab, cond in (("congested", cong), ("calm", ~cong)):
        mask = act & cond
        out[f"nonzero_{lab}"] = float(nz[mask].mean()) if mask.any() else float("nan")
        out[f"slots_{lab}"] = float(mask.mean())
    mask = act & ~hot & hbusy
    out["mean_other_when_hot_busy"] = float(theta[mask].mean()) if mask.any() else float("nan")
    return out


def fmt_theta(t):
    return (f"   CIO on active cells: zero {100 * t['zero_share']:.0f}%  mean|θ| {t['mean_abs']:.2f} dB  "
            f"θ≠0 congested/calm {100 * t['nonzero_congested']:.1f}%/{100 * t['nonzero_calm']:.1f}%  "
            f"hotspot busy {t['mean_hot_busy']:+.2f}  others while hotspot busy "
            f"{t['mean_other_when_hot_busy']:+.2f}")


def fmt(r):
    return (f"{r['method']:20s} s{r['seed']}  ES {r['ES']:5.1f}%  V {r['V']:.4f}  "
            f"ok {r['ok']}/{r['n']}  O {r['O']:.4f}  H {r['H']:.4f}  Xi {r['Xi']:5.1f}  "
            f"short {100 * r['short']:3.0f}%  Jain {r['jain']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mode")
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mlb_off", action="store_true", help="replace a learned MLB by CIO = 0")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = [evaluate(r, a.episodes, greedy=not a.stochastic, ckpt=a.ckpt, mlb_off=a.mlb_off)
           for r in a.runs]
    for r in res:
        print(fmt(r))
        print(fmt_theta(r["theta"]))
    if len(res) > 1:
        print("mean over runs: " + "  ".join(
            f"{k} {np.mean([r[k] for r in res]):.4f}±{np.std([r[k] for r in res]):.4f}"
            for k in ("ES", "V", "H", "Xi")))
