"""PPO trainer for ACCORD and the learned baselines (Sec. IV and V-B).

Every method is the same two-time-scale loop with different flags (METHODS):
who learns, what each actor and critic observes, which reward each agent gets,
and which agents the multiplier binds.

  ippo                [ACCORD] the proposed method (DTDE): separate rewards, lambda on ES,
                      actors and critics on each agent's own observation
  main table (baselines and the centralized reference):
  es_only             ES learns, MLB follows the dead-zone rule
  es_only_i           [ES-only learning] as es_only with the integral multiplier update (38)
  es_cio0_i           [Learned ES, MLB off] as es_cio0 with the integral multiplier update (38)
                      (no learned MLB: equilibrium lambda ~8, too high for the PID gains)
  es_cio0             ES learns, CIO fixed at 0 (MLB off); es_only / es_cio0 use the default PID
  central             [Centralized] reference, not deployable: common reward -(E + w_h H),
                      lambda on both, the MLB head also sees s_t (one agent setting both
                      variables, factorized as pi(s | x) pi(theta | x, s))
  ablation (one design element of ACCORD changed; paper label in brackets):
  accord              [ACCORD (CTDE)] global observations and per-agent critics on x_t
  dtde_team           [ACCORD (team reward)] common reward -(E + w_h H), lambda on both
  dtde_sym            [ACCORD (symmetric lambda)] separate rewards, lambda on both
  dtde_wh0            [ACCORD, w_h = 0] no handover cost in the MLB reward
                      (with lambda on the MLB, the MLB has its own cost critic on o^M)
  older CTDE-based variants: team, accord_sym, accord_no_theta_es, accord_no_tau_mlb, accord_wh0
  multiplier: PID Lagrangian (kp 20, ki 0.1) by default; --lam_ctrl i gives the integral
  update (38), the ablation row ACCORD (integral lambda)
  mlb_allon           sanity check: all cells on, MLB learns (mlb_allon_wh0: w_h = 0)

    .venv/bin/python trainer.py --method accord --iters 500 --seed 0
"""
import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import RuleMLB
from probe import build_configs
from ran_env import PRESETS, EnvConfig, RanEnv

# ------------------------------------------------------------------ methods
BASE = dict(es="learn", mlb="learn", es_reward="energy", mlb_reward="service",
            lam_on=("E",), critic="global", es_obs="global", mlb_obs="global",
            es_drop=(), mlb_drop=(), mlb_sees_s=False, w_h=None, lam_ctrl=None)


def _m(**kw):
    return dict(BASE, **kw)


METHODS = {
    "accord": _m(),
    "es_only": _m(mlb="rule_dz"),
    "ippo": _m(critic="local", es_obs="es_local", mlb_obs="mlb_local"),
    "team": _m(es_reward="team", mlb_reward="team", lam_on=("E", "M")),
    "central": _m(es_reward="team", mlb_reward="team", lam_on=("E", "M"), mlb_sees_s=True),
    "accord_sym": _m(lam_on=("E", "M")),
    "accord_no_theta_es": _m(es_drop=("theta",)),
    "accord_no_tau_mlb": _m(mlb_drop=("tau",)),
    "accord_wh0": _m(w_h=0.0),
    # ablation of ACCORD (DTDE = ippo): one design element changed, local observations kept
    "dtde_team": _m(critic="local", es_obs="es_local", mlb_obs="mlb_local",
                    es_reward="team", mlb_reward="team", lam_on=("E", "M")),
    "dtde_sym": _m(critic="local", es_obs="es_local", mlb_obs="mlb_local", lam_on=("E", "M")),
    "dtde_wh0": _m(critic="local", es_obs="es_local", mlb_obs="mlb_local", w_h=0.0),
    "mlb_allon": _m(es="allon", lam_on=()),
    "mlb_allon_wh0": _m(es="allon", lam_on=(), w_h=0.0),
    "es_cio0": _m(mlb="zero"),
    # main-table baselines without a learned MLB: their equilibrium multiplier is ~8, which the
    # PID gains (kp 20, ki 0.1, set for lambda ~2) do not reach within 1000 iterations, so they
    # use the integral update (38) with eta_lam 0.5
    "es_only_i": _m(mlb="rule_dz", lam_ctrl="i"),
    "es_cio0_i": _m(mlb="zero", lam_ctrl="i"),
    "mappo": _m(critic="global", es_obs="es_local", mlb_obs="mlb_local"),
}


@dataclass
class HP:
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gamma_e: float = 0.98
    lam_gae: float = 0.95
    clip: float = 0.2
    ent: float = 0.01
    ent_es_end: float = 0.0      # linear decay of the ES entropy bonus to this value (None: constant)
    ent_mlb_end: float = None    # same for the MLB; None keeps the bonus constant
    epochs: int = 4
    mb_m: int = 512
    mb_e: int = 64
    hidden: int = 128
    eta_lam: float = 0.5         # Table II's 0.01 let ES collapse to all-off (README)
    lam0: float = 1.0
    lam_max: float = 20.0
    lam_ctrl: str = "pid"        # 'pid' (default): PID Lagrangian (Stooke et al., ICML 2020),
                                 # lambda = kp e + ki I + kd (C_hat increase), I = max(0, I + e),
                                 # e = C_hat - V_max; 'i': lambda += eta_lam e, the integral update
                                 # (38), which overshoots to 5-6 and needs ~1000 iterations (ablation)
    kp: float = 20.0
    ki: float = 0.1
    kd: float = 0.0
    kappa: float = 0.1
    max_grad: float = 0.5
    envs: int = 16
    mlb_action: str = "delta"    # 'delta': -1/0/+1 CIO step (default); 'abs': CIO level as in (25)
    mlb_shaping: str = "none"    # MLB reward: 'none' -V - w_h H (18); 'smooth' replaces V by
                                 # its expectation under load jitter sigma; 'excess' adds
                                 # -shape_w * sum of load above 1/beta
    shape_sigma: float = 0.1
    shape_w: float = 0.05
    v_target: float = None       # constraint level the multiplier trains against (None: V_max)
    mlb_arch: str = "flat"       # 'flat': one MLP to B x |Theta| logits (25); 'shared': the same
                                 # small network on each cell's local features


# ------------------------------------------------------------------ observations
KEYS = {"global": ("rho", "s", "theta", "D", "tau"),
        "es_local": ("rho", "D", "s", "tau"),
        "mlb_local": ("rho", "D", "theta")}


def obs_parts(env):
    """Entries of the global state x_t (14), roughly normalized."""
    cfg = env.cfg
    scale = cfg.demand_mbps * 1e6 * cfg.n_ue / cfg.n_cells
    return dict(rho=np.minimum(env.rho, 3.0), s=env.s.astype(np.float32),
                theta=env.theta / max(-cfg.cio_min, cfg.cio_max),
                D=(env.dem_hist / scale).reshape(env.E, -1), tau=env.tau / cfg.T_dwell)


def build_obs(parts, keys, drop=(), extra=None):
    xs = [parts[k] for k in keys if k not in drop]
    if extra is not None:
        xs.append(extra)
    return np.concatenate(xs, -1).astype(np.float32)


def cell_features(parts, nbr, m, s_next=None):
    """Per-cell local features for the shared MLB actor, (E, B, F): own load,
    activation, offset, dwell timer and demand window; mean and max load of the
    active neighbours; their number; the cluster mean load."""
    local = m["mlb_obs"] == "mlb_local"
    rho, th = parts["rho"], parts["theta"]
    E, B = rho.shape
    act = (rho > 0).astype(np.float32) if local else parts["s"]
    A = nbr.astype(np.float32)
    load = rho * act
    cnt = act @ A.T
    nb_mean = load @ A.T / np.maximum(cnt, 1.0)
    nb_max = np.where(A[None] > 0, load[:, None, :], 0.0).max(-1)
    feats = [rho, th, nb_mean, nb_max, cnt / max(B - 1, 1),
             np.repeat(load.mean(-1, keepdims=True), B, -1)]
    if not local:
        feats.append(parts["s"])
        if "tau" not in m["mlb_drop"]:
            feats.append(parts["tau"])
    if m["mlb_sees_s"] and s_next is not None:
        feats.append(s_next.astype(np.float32))
    D = parts["D"].reshape(E, B, -1)
    return np.concatenate([np.stack(feats, -1), D], -1).astype(np.float32)


class ObsSpec:
    """What every network of a method sees."""

    def __init__(self, m, hp=None, nbr=None):
        self.m = m
        self.shared = hp is not None and hp.mlb_arch == "shared"
        self.nbr = nbr

    def es(self, parts):
        return build_obs(parts, KEYS[self.m["es_obs"]], self.m["es_drop"])

    def mlb(self, parts, s_next):
        if self.shared:
            return cell_features(parts, self.nbr, self.m, s_next)
        extra = s_next.astype(np.float32) if self.m["mlb_sees_s"] else None
        return build_obs(parts, KEYS[self.m["mlb_obs"]], self.m["mlb_drop"], extra)

    def critic_e(self, parts):
        return build_obs(parts, KEYS["global"]) if self.m["critic"] == "global" else self.es(parts)

    def critic_m(self, parts, s_next):
        return build_obs(parts, KEYS["global"]) if self.m["critic"] == "global" \
            else self.mlb(parts, s_next)

    def critic_c(self, parts):
        return self.critic_e(parts)


# ------------------------------------------------------------------ networks
def mlp(i, o, h):
    net = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    return net


class ESActor(nn.Module):
    """Per-cell Bernoulli activation logits (23)-(24); starts near all-on."""

    def __init__(self, d, B, h):
        super().__init__()
        self.net = mlp(d, B, h)
        nn.init.orthogonal_(self.net[-1].weight, 0.01)
        nn.init.constant_(self.net[-1].bias, 3.0)

    def forward(self, x):
        return self.net(x)


class MLBSharedActor(nn.Module):
    """One network applied to every cell's local features -> |Theta| logits per cell."""

    def __init__(self, d, G, h, zero_idx):
        super().__init__()
        self.net = mlp(d, G, h)
        nn.init.orthogonal_(self.net[-1].weight, 0.01)
        bias = torch.zeros(G)
        bias[zero_idx] = 4.0
        with torch.no_grad():
            self.net[-1].bias.copy_(bias)

    def forward(self, x):
        return self.net(x)


class MLBActor(nn.Module):
    """Per-cell categorical over the CIO grid (25); starts near theta = 0."""

    def __init__(self, d, B, G, h, zero_idx):
        super().__init__()
        self.B, self.G = B, G
        self.net = mlp(d, B * G, h)
        nn.init.orthogonal_(self.net[-1].weight, 0.01)
        bias = torch.zeros(B, G)
        bias[:, zero_idx] = 4.0
        with torch.no_grad():
            self.net[-1].bias.copy_(bias.reshape(-1))

    def forward(self, x):
        return self.net(x).reshape(*x.shape[:-1], self.B, self.G)


def es_logp(logits, a, mask):
    lp = -F.binary_cross_entropy_with_logits(logits, a, reduction="none")
    return (lp * mask).sum(-1)


def es_entropy(logits, mask):
    """Mean entropy over the cells allowed to switch."""
    p = torch.sigmoid(logits)
    ent = F.binary_cross_entropy_with_logits(logits, p, reduction="none")
    return (ent * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


def mlb_logp(logits, a):
    return torch.log_softmax(logits, -1).gather(-1, a.unsqueeze(-1)).squeeze(-1).sum(-1)


def mlb_entropy(logits):
    """Mean entropy over cells."""
    lp = torch.log_softmax(logits, -1)
    return -(lp.exp() * lp).sum(-1).mean(-1)


class RunNorm:
    """Running mean/std of the return targets (value normalization)."""

    def __init__(self):
        self.mean, self.var, self.n = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).ravel()
        m, v, n = x.mean(), x.var(), len(x)
        d = m - self.mean
        tot = self.n + n
        self.mean += d * n / tot
        self.var = (self.var * self.n + v * n + d ** 2 * self.n * n / tot) / tot
        self.n = tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


class Nets:
    def __init__(self, m, cfg, hp, env):
        self.nbr = env.nbr
        spec = ObsSpec(m, hp, env.nbr)
        parts = obs_parts(env)
        s0 = env.s
        B = cfg.n_cells
        self.grid = env.theta_grid
        self.dgrid = np.array([-1.0, 0.0, 1.0]) * cfg.cio_step
        G = len(self.grid) if hp.mlb_action == "abs" else 3
        zero_idx = int(np.argmin(np.abs(self.grid))) if hp.mlb_action == "abs" else 1
        h = hp.hidden
        self.es_actor = ESActor(spec.es(parts).shape[-1], B, h) if m["es"] == "learn" else None
        if m["mlb"] != "learn":
            self.mlb_actor = None
        elif hp.mlb_arch == "shared":
            self.mlb_actor = MLBSharedActor(spec.mlb(parts, s0).shape[-1], G, h, zero_idx)
        else:
            self.mlb_actor = MLBActor(spec.mlb(parts, s0).shape[-1], B, G, h, zero_idx)
        self.v_e = mlp(spec.critic_e(parts).shape[-1], 1, h) if m["es"] == "learn" else None
        self.v_m = mlp(spec.critic_m(parts, s0).shape[-1], 1, h) if m["mlb"] == "learn" else None
        self.v_c = mlp(spec.critic_c(parts).shape[-1], 1, h) if m["lam_on"] else None
        # with local critics the MLB cannot use the ES-observation cost critic: its own cost
        # critic on o^M estimates the same return of V_t
        self.v_cm = mlp(spec.critic_m(parts, s0).shape[-1], 1, h) \
            if m["critic"] == "local" and "M" in m["lam_on"] and m["mlb"] == "learn" else None
        self.norm = {k: RunNorm() for k in ("e", "m", "c")}
        self.opt = {}
        for k, net, lr in (("es", self.es_actor, hp.lr_actor), ("mlb", self.mlb_actor, hp.lr_actor),
                           ("v_e", self.v_e, hp.lr_critic), ("v_m", self.v_m, hp.lr_critic),
                           ("v_c", self.v_c, hp.lr_critic), ("v_cm", self.v_cm, hp.lr_critic)):
            if net is not None:
                self.opt[k] = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    def state_dict(self):
        out = {}
        for k in ("es_actor", "mlb_actor", "v_e", "v_m", "v_c", "v_cm"):
            net = getattr(self, k)
            if net is not None:
                out[k] = net.state_dict()
        out["norm"] = {k: vars(v) for k, v in self.norm.items()}
        return out

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k == "norm":
                for kk, vv in v.items():
                    self.norm[kk].__dict__.update(vv)
            elif getattr(self, k) is not None:
                getattr(self, k).load_state_dict(v)


# ------------------------------------------------------------------ action sampling
def sample_es(nets, obs, elig, s_prev, rng, greedy=False):
    with torch.no_grad():
        logits = nets.es_actor(torch.from_numpy(obs))
    p = torch.sigmoid(logits).numpy()
    a = (p > 0.5) if greedy else (rng.random(p.shape) < p)
    a = np.where(elig, a, s_prev > 0).astype(np.float32)
    lp = es_logp(logits, torch.from_numpy(a), torch.from_numpy(elig.astype(np.float32))).numpy()
    return a, lp


def sample_mlb(nets, obs, rng, greedy=False):
    with torch.no_grad():
        logits = nets.mlb_actor(torch.from_numpy(obs))
    probs = torch.softmax(logits, -1).numpy()
    if greedy:
        idx = probs.argmax(-1)
    else:
        u = rng.random(probs.shape[:-1] + (1,))
        idx = np.minimum((np.cumsum(probs, -1) < u).sum(-1), probs.shape[-1] - 1)
    lp = mlb_logp(logits, torch.from_numpy(idx)).numpy()
    return idx, lp


def theta_from(nets, idx, theta_prev, cfg, mode):
    if mode == "abs":
        return nets.grid[idx]
    return np.clip(theta_prev + nets.dgrid[idx], cfg.cio_min, cfg.cio_max)


# ------------------------------------------------------------------ rollout
def rollout(env, m, nets, hp, rng):
    cfg = env.cfg
    spec = ObsSpec(m, hp, env.nbr)
    E, T, K = env.E, cfg.T, cfg.K
    w_h = cfg.w_h if m["w_h"] is None else m["w_h"]
    rule = RuleMLB(trigger=0.9)
    rule.reset(env)
    B = {k: [] for k in ("om", "am", "lpm", "xm", "xc", "V", "H", "Eb", "sw", "on",
                         "oe", "ae", "lpe", "mask", "xe", "th", "rho", "nserv", "out", "act")}
    env.reset()
    for t in range(T):
        parts = obs_parts(env)
        epoch = t % K == 0
        if epoch:
            if m["es"] == "learn":
                o = spec.es(parts)
                elig = env.eligible()
                a, lp = sample_es(nets, o, elig, env.s, rng)
                B["oe"].append(o), B["ae"].append(a), B["lpe"].append(lp)
                B["mask"].append(elig.astype(np.float32))
                B["xe"].append(spec.critic_e(parts))
                s_req = a.astype(int)
            else:
                s_req = np.ones_like(env.s)
            s_next = np.where(env.eligible(), s_req, env.s)
        else:
            s_req = s_next = env.s
        if m["mlb"] == "learn":
            o = spec.mlb(parts, s_next)
            idx, lp = sample_mlb(nets, o, rng)
            theta = theta_from(nets, idx, env.theta, cfg, hp.mlb_action)
            B["om"].append(o), B["am"].append(idx), B["lpm"].append(lp)
            B["xm"].append(spec.critic_m(parts, s_next))
        elif m["mlb"] == "rule_dz":
            theta = rule.act(env, s_next)
        else:
            theta = np.zeros_like(env.theta)
        if nets.v_c is not None:
            B["xc"].append(spec.critic_c(parts))
        _, info, _ = env.step(s_req, theta)
        B["V"].append(info["V"]), B["H"].append(info["H"]), B["Eb"].append(info["Ebar"])
        B["sw"].append(info["switches"]), B["on"].append(info["s"].sum(-1))
        B["th"].append(np.where(info["s"] > 0, info["theta"], np.nan))
        B["rho"].append(info["rho"]), B["nserv"].append(info["n_serv"])
        B["out"].append(info["out"]), B["act"].append(info["s"] > 0)
    parts = obs_parts(env)
    last = dict(xe=spec.critic_e(parts) if m["es"] == "learn" else None,
                xm=spec.critic_m(parts, env.s) if m["mlb"] == "learn" else None,
                xc=spec.critic_c(parts) if nets.v_c is not None else None)
    out = {k: np.array(v) for k, v in B.items() if len(v)}
    V, H, Eb = out["V"], out["H"], out["Eb"]                 # (T, E)
    out["rm"] = -mlb_service(out, cfg, hp) - w_h * H if m["mlb_reward"] == "service" \
        else -(Eb + w_h * H)
    per_slot = Eb if m["es_reward"] == "energy" else Eb + w_h * H
    out["re"] = -per_slot.reshape(T // K, K, E).mean(1)      # (T/K, E)
    return out, last


def mlb_service(out, cfg, hp):
    """Service term of the MLB reward, (T, E)."""
    V = out["V"]
    if hp.mlb_shaping == "none":
        return V
    lim = 1.0 / cfg.beta
    rho, act = out["rho"], out["act"]
    if hp.mlb_shaping == "excess":
        return V + hp.shape_w * (np.maximum(rho - lim, 0.0) * act).sum(-1)
    # 'smooth': outage plus the expected share of UEs on cells pushed above 1/beta
    # when the load fluctuates by sigma
    p_over = torch.distributions.Normal(0.0, 1.0).cdf(
        torch.from_numpy((rho - lim) / hp.shape_sigma)).numpy()
    return out["out"] + (out["nserv"] / cfg.n_ue * p_over * act).sum(-1)


def gae(r, v, v_last, gamma, lam):
    adv = np.zeros_like(r)
    nxt = 0.0
    for t in reversed(range(len(r))):
        v_next = v_last if t == len(r) - 1 else v[t + 1]
        delta = r[t] + gamma * v_next - v[t]
        nxt = delta + gamma * lam * nxt
        adv[t] = nxt
    return adv, adv + v


def values(net, norm, x):
    with torch.no_grad():
        y = net(torch.from_numpy(x)).squeeze(-1).numpy()
    return y * norm.std + norm.mean


def normed(a):
    return (a - a.mean()) / (a.std() + 1e-8)


# ------------------------------------------------------------------ update
def entropy_coefs(hp, it, iters):
    """Entropy coefficients of the ES and MLB actors at iteration it (linear decay)."""
    f = (it - 1) / max(iters - 1, 1)
    es = hp.ent if hp.ent_es_end is None else hp.ent + f * (hp.ent_es_end - hp.ent)
    mlb = hp.ent if hp.ent_mlb_end is None else hp.ent + f * (hp.ent_mlb_end - hp.ent)
    return es, mlb


def update(nets, m, hp, data, last, state, rng, ent_es=None, ent_mlb=None):
    cfg_K = data["re"].shape[0]
    T, E = data["V"].shape
    K = T // cfg_K
    logs = {}
    # dual update (38)
    v_now = data["V"].mean()
    state["C_hat"] = v_now if state["C_hat"] is None else \
        (1 - hp.kappa) * state["C_hat"] + hp.kappa * v_now
    if m["lam_on"] and hp.lam_ctrl == "pid":
        e = state["C_hat"] - state["v_max"]
        state["I"] = max(0.0, state["I"] + e)
        rise = 0.0 if state["C_prev"] is None else max(0.0, state["C_hat"] - state["C_prev"])
        state["C_prev"] = state["C_hat"]
        state["lam"] = float(np.clip(hp.kp * e + hp.ki * state["I"] + hp.kd * rise, 0.0, hp.lam_max))
    elif m["lam_on"]:
        state["lam"] = float(np.clip(state["lam"] + hp.eta_lam * (state["C_hat"] - state["v_max"]),
                                     0.0, hp.lam_max))
    lam = state["lam"]
    # cost advantages (33)-(34) and the epoch average (37)
    if nets.v_c is not None:
        vc = values(nets.v_c, nets.norm["c"], data["xc"])
        vc_last = values(nets.v_c, nets.norm["c"], last["xc"])
        A_c, G_c = gae(data["V"], vc, vc_last, hp.gamma, hp.lam_gae)
        A_c_e = A_c.reshape(cfg_K, K, E).mean(1)
    A_cm = None
    if nets.v_cm is not None:
        vcm = values(nets.v_cm, nets.norm["c"], data["xm"])
        vcm_last = values(nets.v_cm, nets.norm["c"], last["xm"])
        A_cm, G_cm = gae(data["V"], vcm, vcm_last, hp.gamma, hp.lam_gae)
    # MLB (31)-(32)
    if nets.mlb_actor is not None:
        vm = values(nets.v_m, nets.norm["m"], data["xm"])
        vm_last = values(nets.v_m, nets.norm["m"], last["xm"])
        A_m, G_m = gae(data["rm"], vm, vm_last, hp.gamma, hp.lam_gae)
        Ahat_m = normed(A_m)
        if "M" in m["lam_on"]:
            Ahat_m = (Ahat_m - lam * normed(A_c if A_cm is None else A_cm)) / (1 + lam)
    # ES (35)-(36), penalized advantage (39)
    if nets.es_actor is not None:
        ve = values(nets.v_e, nets.norm["e"], data["xe"])
        ve_last = values(nets.v_e, nets.norm["e"], last["xe"])
        A_e, G_e = gae(data["re"], ve, ve_last, hp.gamma_e, hp.lam_gae)
        Ahat_e = normed(A_e)
        if "E" in m["lam_on"]:
            Ahat_e = (Ahat_e - lam * normed(A_c_e)) / (1 + lam)
    for k, G in (("m", G_m if nets.mlb_actor is not None else None),
                 ("e", G_e if nets.es_actor is not None else None),
                 ("c", G_c if nets.v_c is not None else None)):
        if G is not None:
            nets.norm[k].update(G)

    def flat(x):
        return torch.from_numpy(np.ascontiguousarray(x.reshape(-1, *x.shape[2:])))

    def critic_step(name, net, x, G, idx, norm):
        tgt = (G[idx] - norm.mean) / norm.std
        loss = F.mse_loss(net(x[idx]).squeeze(-1), tgt)
        nets.opt[name].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), hp.max_grad)
        nets.opt[name].step()
        return loss.item()

    # slot-scale updates: MLB actor, MLB critic, cost critic
    n_slot = T * E
    if nets.mlb_actor is not None or nets.v_c is not None:
        tens = {}
        if nets.mlb_actor is not None:
            tens.update(om=flat(data["om"]), am=flat(data["am"]).long(), lpm=flat(data["lpm"]),
                        A=flat(Ahat_m.astype(np.float32)), xm=flat(data["xm"]),
                        Gm=flat(G_m.astype(np.float32)))
        if nets.v_c is not None:
            tens.update(xc=flat(data["xc"]), Gc=flat(G_c.astype(np.float32)))
        if nets.v_cm is not None:
            tens.update(Gcm=flat(G_cm.astype(np.float32)))
        lm, lc, lv, ent_m = [], [], [], []
        for _ in range(hp.epochs):
            perm = torch.from_numpy(rng.permutation(n_slot))
            for i in range(0, n_slot, hp.mb_m):
                idx = perm[i:i + hp.mb_m]
                if nets.mlb_actor is not None:
                    logits = nets.mlb_actor(tens["om"][idx])
                    lp = mlb_logp(logits, tens["am"][idx])
                    ratio = torch.exp(lp - tens["lpm"][idx])
                    A = tens["A"][idx]
                    ent = mlb_entropy(logits).mean()
                    loss = -torch.min(ratio * A, torch.clamp(ratio, 1 - hp.clip, 1 + hp.clip) * A).mean() \
                        - ent_mlb * ent
                    nets.opt["mlb"].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(nets.mlb_actor.parameters(), hp.max_grad)
                    nets.opt["mlb"].step()
                    lm.append(loss.item()), ent_m.append(ent.item())
                    lv.append(critic_step("v_m", nets.v_m, tens["xm"], tens["Gm"], idx, nets.norm["m"]))
                    if nets.v_cm is not None:
                        critic_step("v_cm", nets.v_cm, tens["xm"], tens["Gcm"], idx, nets.norm["c"])
                if nets.v_c is not None:
                    lc.append(critic_step("v_c", nets.v_c, tens["xc"], tens["Gc"], idx, nets.norm["c"]))
        logs.update(loss_mlb=np.mean(lm) if lm else np.nan, ent_mlb=np.mean(ent_m) if ent_m else np.nan,
                    loss_vm=np.mean(lv) if lv else np.nan, loss_vc=np.mean(lc) if lc else np.nan)
    # epoch-scale updates: ES actor and critic
    if nets.es_actor is not None:
        oe, ae, lpe, mask = flat(data["oe"]), flat(data["ae"]), flat(data["lpe"]), flat(data["mask"])
        Ae, xe, Ge = flat(Ahat_e.astype(np.float32)), flat(data["xe"]), flat(G_e.astype(np.float32))
        n_ep = oe.shape[0]
        le, lve, ent_e = [], [], []
        for _ in range(hp.epochs):
            perm = torch.from_numpy(rng.permutation(n_ep))
            for i in range(0, n_ep, hp.mb_e):
                idx = perm[i:i + hp.mb_e]
                logits = nets.es_actor(oe[idx])
                lp = es_logp(logits, ae[idx], mask[idx])
                ratio = torch.exp(lp - lpe[idx])
                A = Ae[idx]
                ent = es_entropy(logits, mask[idx]).mean()
                loss = -torch.min(ratio * A, torch.clamp(ratio, 1 - hp.clip, 1 + hp.clip) * A).mean() \
                    - ent_es * ent
                nets.opt["es"].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(nets.es_actor.parameters(), hp.max_grad)
                nets.opt["es"].step()
                le.append(loss.item()), ent_e.append(ent.item())
                lve.append(critic_step("v_e", nets.v_e, xe, Ge, idx, nets.norm["e"]))
        logs.update(loss_es=np.mean(le), ent_es=np.mean(ent_e), loss_ve=np.mean(lve))
    return logs


# ------------------------------------------------------------------ main
def train(method, cfg_kw, hp, iters, seed, out, save_every=50, log_every=1):
    m = METHODS[method]
    if m["lam_ctrl"] is not None:                    # the method fixes its multiplier update
        hp = replace(hp, lam_ctrl=m["lam_ctrl"])
    cfg = EnvConfig(**cfg_kw)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 12345)
    env = RanEnv(cfg, hp.envs, seed=10_000_000 + 100_000 * seed)
    env.reset()
    nets = Nets(m, cfg, hp, env)
    json.dump(dict(method=method, spec=m, cfg=cfg_kw, hp=asdict(hp), seed=seed),
              open(os.path.join(out, "config.json"), "w"), indent=1, default=list)
    state = dict(lam=hp.lam0 if m["lam_on"] else 0.0, C_hat=None,
                 v_max=cfg.v_max if hp.v_target is None else hp.v_target,
                 I=hp.lam0 / hp.ki if hp.ki > 0 else 0.0, C_prev=None)   # PID: starts at lam0
    fields = ["iter", "sec", "V", "C_hat", "lam", "Ebar", "H", "n_on", "switches",
              "th_zero", "th_abs", "R_E", "R_M", "loss_es", "loss_mlb", "loss_ve", "loss_vm",
              "loss_vc", "ent_es", "ent_mlb"]
    logf = open(os.path.join(out, "log.csv"), "w", newline="")
    writer = csv.DictWriter(logf, fieldnames=fields)
    writer.writeheader()
    t0 = time.time()
    for it in range(1, iters + 1):
        env = RanEnv(cfg, hp.envs, seed=10_000_000 + 100_000 * seed + it)
        data, last = rollout(env, m, nets, hp, rng)
        ent_es, ent_mlb = entropy_coefs(hp, it, iters)
        logs = update(nets, m, hp, data, last, state, rng, ent_es, ent_mlb)
        row = dict(iter=it, sec=round(time.time() - t0, 1), V=data["V"].mean(), C_hat=state["C_hat"],
                   lam=state["lam"], Ebar=data["Eb"].mean(), H=data["H"].mean(),
                   n_on=data["on"].mean(), switches=data["sw"].sum(0).mean(),
                   th_zero=float((data["th"][np.isfinite(data["th"])] == 0).mean()),
                   th_abs=float(np.nanmean(np.abs(data["th"]))),
                   R_E=data["re"].mean(), R_M=data["rm"].mean(), **logs)
        writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                         if k in fields})
        logf.flush()
        if it % log_every == 0:
            print(f"[{method} s{seed}] it {it:4d}  V {row['V']:.4f}  C {row['C_hat']:.4f}  "
                  f"lam {row['lam']:.3f}  Ebar {row['Ebar']:.3f}  on {row['n_on']:.2f}  "
                  f"H {row['H']:.4f}  sw {row['switches']:.1f}  th0 {row['th_zero']:.2f}  "
                  f"{row['sec']:.0f}s", flush=True)
        if it % save_every == 0 or it == iters:
            torch.save(dict(nets=nets.state_dict(), state=state, iter=it),
                       os.path.join(out, f"ckpt_{it}.pt" if it != iters else "final.pt"))
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="accord", choices=sorted(METHODS))
    ap.add_argument("--preset", default="hotspot", choices=sorted(PRESETS))
    ap.add_argument("--base", default="", help="env overrides k=v,k=v")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mlb_action", default="delta", choices=["abs", "delta"])
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eta_lam", type=float, default=None)
    ap.add_argument("--lam0", type=float, default=None)
    ap.add_argument("--ent", type=float, default=None)
    ap.add_argument("--ent_es_end", type=float, default=None, help="negative: no ES decay")
    ap.add_argument("--ent_mlb_end", type=float, default=None)
    ap.add_argument("--mlb_shaping", default=None, choices=["none", "smooth", "excess"])
    ap.add_argument("--mlb_arch", default=None, choices=["flat", "shared"])
    ap.add_argument("--v_target", type=float, default=None,
                    help="train the multiplier against this level instead of V_max")
    ap.add_argument("--shape_sigma", type=float, default=None)
    ap.add_argument("--shape_w", type=float, default=None)
    ap.add_argument("--lam_ctrl", default=None, choices=["i", "pid"])
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--ki", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg_kw = next(iter(build_configs(a.preset, a.base, []).values()))
    hp = HP(envs=a.envs, mlb_action=a.mlb_action)
    for k in ("eta_lam", "lam0", "ent", "ent_es_end", "ent_mlb_end", "mlb_shaping",
              "shape_sigma", "shape_w", "mlb_arch", "v_target", "lam_ctrl", "kp", "ki", "kd"):
        if getattr(a, k) is not None:
            setattr(hp, k, getattr(a, k))
    for k in ("ent_es_end", "ent_mlb_end"):          # a negative end value: no decay
        if getattr(hp, k) is not None and getattr(hp, k) < 0:
            setattr(hp, k, None)
    out = a.out or f"runs/{a.method}_s{a.seed}"
    train(a.method, cfg_kw, hp, a.iters, a.seed, out, a.save_every)
