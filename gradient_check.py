"""Does the MLB reward have a slope in the direction of a CIO step?

All cells on; snapshots where the most loaded cell exceeds 1.0.  The offset of
that cell is set to each value in STEPS (others 0) and, for every variant, the
true V, the smoothed V (sigma), the overload excess and the share of UEs whose
serving cell changes relative to theta = 0 (a one-off handover) are measured.

    .venv/bin/python gradient_check.py
"""
import numpy as np
import torch

from baselines import fixed_point
from ran_env import RanEnv, evaluate, make_config

STEPS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def measures(cfg, ev, s, sigma):
    U, B = ev["serv"].shape[-1], s.shape[-1]
    served = ~ev["outage"]
    n = np.stack([(served & (ev["serv"] == b)).sum(-1) for b in range(B)], -1)
    lim = 1.0 / cfg.beta
    act = s > 0
    p = torch.distributions.Normal(0.0, 1.0).cdf(
        torch.from_numpy((ev["rho"] - lim) / sigma)).numpy()
    smooth = ev["out_frac"] + (n / U * p * act).sum(-1)
    excess = (np.maximum(ev["rho"] - lim, 0.0) * act).sum(-1)
    return ev["V"], smooth, excess


def check(preset, n_envs=10, every=10, sigma=0.1, seed=7, w_h=0.1):
    cfg = make_config(preset)
    env = RanEnv(cfg, n_envs, seed)
    env.reset()
    ts = np.arange(0, cfg.T, every)
    rsrp = env.rsrp_at(ts).reshape(-1, cfg.n_ue, cfg.n_cells)         # (N, U, B)
    d = env.demand_at(ts).reshape(-1, cfg.n_ue)
    N, B = rsrp.shape[0], cfg.n_cells
    s = np.ones((N, B), int)
    rho0 = fixed_point(cfg, rsrp, d, s, np.zeros((N, B)))
    ev0 = evaluate(cfg, rsrp, d, s, np.zeros((N, B)), rho0)
    b_star = ev0["rho"].argmax(-1)
    pick = ev0["rho"].max(-1) > 1.0
    rsrp, d, s, b_star, rho0 = rsrp[pick], d[pick], s[pick], b_star[pick], rho0[pick]
    M = len(b_star)
    theta = np.zeros((M, len(STEPS), B))
    theta[np.arange(M), :, b_star] = STEPS
    rsrp_k, d_k, s_k = rsrp[:, None], d[:, None], s[:, None]
    rho = fixed_point(cfg, rsrp_k, d_k, s_k, theta, rho0[:, None])
    ev = evaluate(cfg, rsrp_k, d_k, s_k, theta, rho)
    V, Vs, X = measures(cfg, ev, s_k, sigma)                           # (M, K)
    k0 = int(np.where(STEPS == 0)[0][0])
    moved = (ev["serv"] != ev["serv"][:, k0:k0 + 1]).mean(-1)          # (M, K)
    print(f"\n{preset}: {M} congested snapshots out of {N} "
          f"(load of the congested cell {ev0['rho'].max(-1)[pick].mean():.2f} on average)")
    print(f"{'theta':>6s} {'V':>8s} {'smooth V':>9s} {'excess':>8s} {'moved UEs':>10s}  "
          f"{'P(smooth V lower than at 0)':>28s}")
    for k, st in enumerate(STEPS):
        lower = np.mean(Vs[:, k] < Vs[:, k0] - 1e-9) if k != k0 else np.nan
        print(f"{st:+6.0f} {V[:, k].mean():8.4f} {Vs[:, k].mean():9.4f} {X[:, k].mean():8.4f} "
              f"{moved[:, k].mean():10.4f}  {lower:28.2f}")
    km1 = int(np.where(STEPS == -1)[0][0])
    dV, dVs, dH = V[:, km1] - V[:, k0], Vs[:, km1] - Vs[:, k0], moved[:, km1]
    print(f"theta = -1 vs 0: mean dV {dV.mean():+.4f}, mean d(smooth V) {dVs.mean():+.4f} per slot; "
          f"one-off handover cost w_h * moved = {w_h * dH.mean():.4f}; "
          f"share of snapshots where the step changes smooth V by > 0.001: "
          f"{np.mean(np.abs(dVs) > 1e-3):.2f}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    for preset in ("hotspot", "hotspot_tight"):
        check(preset)
