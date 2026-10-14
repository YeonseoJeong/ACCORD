"""Non-learning controllers used to probe the environment before training ACCORD.

ES policies act at epoch starts and return a requested activation (the env
applies the dwell mask); MLB policies act every slot and return CIO offsets.

  AllOnES / ZeroMLB   : All-on reference (Sec. V-B)
  RuleES / RuleMLB    : threshold SON pair of Sec. V-B
  CioOracleMLB        : CIO re-optimised on the true per-UE channels every few
                        slots, keeping a load margin (what the CIO can achieve)
  OracleES            : centralized lookahead on the true future channels and
                        demand (a privileged-information reference for a learned
                        ES; a heuristic, so not a strict upper bound)

Because one cell can hold a large share of the UEs (hotspots) and V counts all
UEs of a cell once its load passes 1/beta, V jumps when a loaded cell crosses
that line while the load itself fluctuates from slot to slot.  The oracles
therefore plan against 1/beta - margin and score overload by its excess.
"""
import itertools

import numpy as np

from ran_env import RanEnv, evaluate


# ------------------------------------------------------------------ helpers
def nbr_mean(rho, act, nbr):
    """Mean load over active neighbors, and the number of active neighbors."""
    A = nbr.astype(float)
    cnt = act @ A.T
    tot = (rho * act) @ A.T
    return tot / np.maximum(cnt, 1), cnt


def rule_step(theta, rho, act, nbr, thr, step, lo, hi):
    """One MLB rule update: move CIO of active cells away from the neighbor mean."""
    m, cnt = nbr_mean(rho, act, nbr)
    diff = rho - m
    upd = (act > 0) & (cnt > 0)
    th = theta - step * (upd & (diff > thr)) + step * (upd & (diff < -thr))
    return np.clip(th, lo, hi)


def plan_cfg(cfg, margin):
    """Config whose V counts a cell as degraded above 1/beta - margin."""
    return cfg.with_(beta=1.0 / (1.0 / cfg.beta - margin)) if margin else cfg


def excess(ev, s, cfg):
    """Total load above 1/beta of cfg over active cells, (...,)."""
    return (np.maximum(ev["rho"] - 1.0 / cfg.beta, 0.0) * (s > 0)).sum(-1)


def fixed_point(cfg, rsrp, d, s, theta, rho=None, n_it=3):
    """Load-coupled interference fixed point."""
    shape = np.broadcast_shapes(np.shape(s), rsrp.shape[:-2] + (rsrp.shape[-1],))
    rho = np.full(shape, 0.5) if rho is None else np.broadcast_to(rho, shape).copy()
    for _ in range(n_it):
        rho = evaluate(cfg, rsrp, d, s, theta, rho)["rho"]
    return rho


def rule_fixed_point(cfg, nbr, rsrp, d, s, rho_int, thr, n_it=12):
    """CIO that RuleMLB reaches after n_it slots from theta = 0 (no best-keeping)."""
    th = np.zeros(np.broadcast_shapes(s.shape, rho_int.shape))
    act = (s > 0).astype(float)
    for _ in range(n_it):
        ev = evaluate(cfg, rsrp, d, s, th, rho_int)
        th = rule_step(th, ev["rho"], act, nbr, thr, cfg.cio_step, cfg.cio_min, cfg.cio_max)
    return th, evaluate(cfg, rsrp, d, s, th, rho_int)


def cd_cio(cfg, rsrp, d, s, margin=0.15, theta0=None, rho_int=None, rounds=2, w=0.05):
    """Coordinate descent over the CIO grid, every leading index at once.

    Minimises V (with the margin) + w * overload excess; the excess term lets
    the search leave the plateaus of the step-shaped V.  rsrp (..., U, B),
    d (..., U), s (..., B).  Returns theta and the evaluation under plan_cfg.
    """
    pc = plan_cfg(cfg, margin)
    grid = np.arange(cfg.cio_min, cfg.cio_max + 1e-9, cfg.cio_step)
    B = rsrp.shape[-1]
    shape = np.broadcast_shapes(np.shape(s), rsrp.shape[:-2] + (B,))
    act = np.broadcast_to(np.asarray(s) > 0, shape)
    th = np.zeros(shape)
    if theta0 is not None:
        th = np.where(act, np.broadcast_to(theta0, shape), 0.0)
    rho = fixed_point(pc, rsrp, d, s, th, rho_int, n_it=2 if rho_int is not None else 3)

    def score(ev, a):
        return ev["V"] + w * (np.maximum(ev["rho"] - 1.0 / pc.beta, 0.0) * a).sum(-1)

    best = score(evaluate(pc, rsrp, d, s, th, rho), act)
    for _ in range(rounds):
        for b in range(B):
            trial = np.repeat(th[..., None, :], len(grid), axis=-2)
            trial[..., b] = grid
            ev = evaluate(pc, rsrp[..., None, :, :], d[..., None, :],
                          np.asarray(s)[..., None, :], trial, rho[..., None, :])
            sc = score(ev, act[..., None, :]) + 1e-6 * np.abs(grid)
            k = np.argmin(sc, axis=-1)
            v = np.take_along_axis(sc, k[..., None], -1)[..., 0]
            take = (v < best - 1e-9) & act[..., b]
            th[..., b] = np.where(take, grid[k], th[..., b])
            best = np.where(take, v, best)
        rho = evaluate(pc, rsrp, d, s, th, rho)["rho"]
    return th, evaluate(pc, rsrp, d, s, th, rho)


# ------------------------------------------------------------------ ES side
class AllOnES:
    name = "allon"

    def reset(self, env):
        pass

    def observe(self, env, info):
        pass

    def act(self, env):
        return np.ones_like(env.s)


class RuleES:
    """Deactivate a cell whose load stayed below rho_off for an epoch; reactivate
    a sleeping cell when an active neighbor exceeded rho_on (Sec. V-B).

    guard: a cell is deactivated only if every sleeping cell keeps an active
    neighbor, which prevents the absorbing all-off state of the literal rule.
    """

    def __init__(self, rho_off=0.2, rho_on=0.8, guard=True):
        self.rho_off, self.rho_on, self.guard = rho_off, rho_on, guard
        self.name = f"rule(off={rho_off},on={rho_on})"

    def reset(self, env):
        self.ep_max = np.full_like(env.rho, np.inf)     # no full epoch observed yet
        self.n = 0
        self.n_deact = 0          # deactivations made
        self.n_guard = 0          # deactivations given up because of the coverage guard

    def observe(self, env, info):
        self.ep_max = info["rho"].copy() if self.n == 0 else np.maximum(self.ep_max, info["rho"])
        self.n += 1

    def act(self, env):
        s = env.s.copy()
        elig = env.eligible()
        nbr = env.nbr
        rho = self.ep_max
        for e in range(env.E):
            act = s[e] > 0
            hot = (rho[e] > self.rho_on) & act
            for b in range(env.B):
                if not act[b] and elig[e, b] and (nbr[b] & hot).any():
                    s[e, b] = 1
            for b in np.argsort(rho[e]):
                if s[e, b] and env.s[e, b] and elig[e, b] and rho[e, b] < self.rho_off:
                    trial = s[e].copy()
                    trial[b] = 0
                    if self.guard and not _covered(trial, nbr):
                        self.n_guard += 1
                        continue
                    s[e] = trial
                    self.n_deact += 1
        self.n = 0
        return s


def _covered(s, nbr):
    on = s > 0
    if not on.any():
        return False
    return all((nbr[b] & on).any() for b in np.where(~on)[0])


class OracleES:
    """Centralized lookahead on the true future channels and demand.

    Every epoch, candidate activations (the current one and up to max_flips
    changes among eligible cells) are scored over the T_dwell slots that a
    switch commits (offsets 0, K/2, K, 2K, ..., T_dwell - 1 by default) with
    the CIO given by mlb_model ('zero', 'rule' = RuleMLB's settled state, or
    'cio' = cd_cio).  A candidate is admissible if, at every sampled slot, it
    adds no overload above 1/beta - margin and no outage beyond the safest
    reachable candidate (the most cells on); the least-energy admissible one is
    taken.  The margin is the knob that trades energy against V.
    """

    def __init__(self, margin=0.1, mlb_model="cio", max_flips=2,
                 offsets=None, tol=0.0, rule_thr=0.2, cio_w=0.05):
        self.margin, self.mlb_model, self.max_flips = margin, mlb_model, max_flips
        self.offsets, self.tol, self.rule_thr, self.cio_w = offsets, tol, rule_thr, cio_w
        self.name = f"oracle({mlb_model},m={margin})"

    def reset(self, env):
        pass

    def observe(self, env, info):
        pass

    def act(self, env):
        cfg = env.cfg
        pc = plan_cfg(cfg, self.margin)
        offsets = self.offsets
        if offsets is None:   # cover every slot a switch locks: [t, t + T_dwell - 1]
            offsets = sorted({0, cfg.K // 2, cfg.T_dwell - 1} | set(range(cfg.K, cfg.T_dwell, cfg.K)))
        ts = np.array([env.t + o for o in offsets if env.t + o < cfg.T])
        n_first = int((ts < env.t + cfg.K).sum())
        out = env.s.copy()
        elig = env.eligible()
        self.nbr = env.nbr
        for e in range(env.E):
            cand = self._candidates(env.s[e], elig[e])
            rsrp = env.rsrp_at(ts, e)[None]                     # (1, n_t, U, B)
            d = env.demand_at(ts, e)[None]                      # (1, n_t, U)
            S = cand[:, None, :]                                # (n_c, 1, B)
            if self.mlb_model == "cio":
                _, ev = cd_cio(cfg, rsrp, d, S, self.margin, w=self.cio_w)
            else:
                th = np.zeros(np.broadcast_shapes(S.shape, (1, len(ts), cfg.n_cells)))
                rho = fixed_point(pc, rsrp, d, S, th)
                if self.mlb_model == "rule":
                    _, ev = rule_fixed_point(pc, env.nbr, rsrp, d, S, rho, self.rule_thr)
                else:
                    ev = evaluate(pc, rsrp, d, S, th, rho)
            exc = excess(ev, S, pc)                             # (n_c, n_t)
            outage = ev["out_frac"]
            safe = int(np.argmax(cand.sum(1)))
            ok = ((exc <= exc[safe] + self.tol) & (outage <= outage[safe] + 1e-9)).all(1)
            n_sw = (cand != env.s[e]).sum(1)
            energy = ev["E"][:, :n_first].mean(1) + cfg.zeta * n_sw / cfg.K
            out[e] = cand[int(np.argmin(np.where(ok, energy + 1e-6 * n_sw, np.inf)))]
        return out

    def _candidates(self, s, elig):
        idx = np.where(elig)[0]
        cands = [s.copy()]
        for k in range(1, self.max_flips + 1):
            for comb in itertools.combinations(idx, k):
                c = s.copy()
                c[list(comb)] = 1 - c[list(comb)]
                if c.any() and _covered(c, self.nbr):
                    cands.append(c)
        return np.array(cands)


# ------------------------------------------------------------------ MLB side
class ZeroMLB:
    name = "cio0"

    def reset(self, env):
        pass

    def observe(self, env, info):
        pass

    def act(self, env, s_next):
        return np.zeros_like(env.theta)


class RuleMLB:
    """Shift theta_b by -/+delta when cell b's load differs from the mean of its
    active neighbors by more than thr (Sec. V-B); acts on the last slot's loads.

    trigger: optional dead zone.  The rule then acts only in slots where some
    active cell is loaded above `trigger` and otherwise relaxes the offsets
    toward 0, since moving UEs onto weaker cells costs PRBs when nothing is
    congested.
    """

    def __init__(self, thr=0.2, trigger=None):
        self.thr, self.trigger = thr, trigger
        self.name = f"mlbrule({thr}{f',trig={trigger}' if trigger else ''})"

    def reset(self, env):
        pass

    def observe(self, env, info):
        pass

    def act(self, env, s_next):
        cfg = env.cfg
        act = (env.s > 0).astype(float)
        th = rule_step(env.theta, env.rho, act, env.nbr,
                       self.thr, cfg.cio_step, cfg.cio_min, cfg.cio_max)
        if self.trigger is None:
            return th
        busy = (env.rho * act).max(-1, keepdims=True) >= self.trigger
        relax = env.theta - np.sign(env.theta) * cfg.cio_step
        return np.where(busy, th, relax)


class CioOracleMLB:
    """Re-optimise the CIO with cd_cio on the current per-UE channels every
    `every` slots and whenever the observed activation changes, warm-started
    from the offsets in force; holds them in between.

    sees: 'prev' (default) optimises for s_{t-1}, as in the paper's
    simultaneous-move game where both applications act on x_t; a deactivation
    is then seen one slot late.  'next' optimises for the activation applied
    in this slot (sequential: the rApp acts first, the xApp responds).
    """

    def __init__(self, margin=0.15, every=5, rounds=2, w=0.05, sees="prev"):
        self.margin, self.every, self.rounds, self.w, self.sees = margin, every, rounds, w, sees
        self.name = f"cio_oracle(m={margin},every={every},w={w},sees={sees})"

    def reset(self, env):
        self.s_seen = None

    def observe(self, env, info):
        pass

    def act(self, env, s_next):
        s_obs = s_next if self.sees == "next" else env.s
        changed = self.s_seen is None or (s_obs != self.s_seen).any()
        self.s_seen = s_obs.copy()
        if env.t % self.every and not changed:
            return env.theta
        th, _ = cd_cio(env.cfg, env.rsrp_at(env.t), env.demand_at(env.t), s_obs,
                       self.margin, theta0=env.theta, rho_int=env.rho, rounds=self.rounds,
                       w=self.w)
        return th


# ------------------------------------------------------------------ runner
def off_periods(s, cfg):
    """Deactivations and short-lived ones (cell back on within T_dwell + K slots),
    the signature of the activation oscillation of Proposition 1.  s: (T, E, B)."""
    T, E, B = s.shape
    n_off = np.zeros(E)
    n_short = np.zeros(E)
    for e in range(E):
        for b in range(B):
            dx = np.diff(s[:, e, b].astype(int))
            offs, ons = np.where(dx == -1)[0], np.where(dx == 1)[0]
            n_off[e] += len(offs)
            for o in offs:
                later = ons[ons > o]
                if len(later) and later[0] - o <= cfg.T_dwell + cfg.K:
                    n_short[e] += 1
    return dict(n_off=n_off, n_short=n_short)


def run(cfg, es, mlb, n_envs=3, seed=0, record=False):
    """Both applications act on the same state; mlb.act also receives s_t so
    that sequential variants can be studied (CioOracleMLB(sees='next')).
    Policies that follow the paper's simultaneous moves ignore it."""
    env = RanEnv(cfg, n_envs, seed)
    env.reset()
    es.reset(env)
    mlb.reset(env)
    keys = ["E", "Ebar", "V", "H", "out", "switches"]
    log = {k: [] for k in keys}
    trace = {"n_on": [], "rho": [], "theta": [], "s": []}
    s_log = []
    jain = []
    done = False
    while not done:
        s_req = es.act(env) if env.is_epoch_start() else env.s
        s_next = np.where(env.eligible(), s_req, env.s) if env.is_epoch_start() else env.s
        theta = mlb.act(env, s_next)
        _, info, done = env.step(s_req, theta)
        es.observe(env, info)
        mlb.observe(env, info)
        for k in keys:
            log[k].append(info[k])
        act = info["s"] > 0
        s_log.append(act)
        r = info["rho"] * act
        jain.append(r.sum(-1) ** 2 / np.maximum(act.sum(-1) * (r ** 2).sum(-1), 1e-12))
        if record:
            trace["n_on"].append(act.sum(-1))
            trace["rho"].append(info["rho"])
            trace["theta"].append(info["theta"])
            trace["s"].append(info["s"])
    L = {k: np.array(v) for k, v in log.items()}            # (T, E)
    res = dict(E_sum=L["E"].sum(0), V=L["V"].mean(0), H=L["H"].mean(0),
               O=L["out"].mean(0), Xi=L["switches"].sum(0),
               jain=np.array(jain).mean(0), hot_cells=env.hot_cells)
    res.update(off_periods(np.array(s_log), cfg))
    if record:
        res["trace"] = {k: np.array(v) for k, v in trace.items()}
        res["xi"] = env.xi_at(np.arange(cfg.T))
    return res
