"""ES/MLB coexistence environment of "Learning to Coexist" (Sec. II and V-A).

The simulator is vectorized over ``n_envs`` independent episodes.  UE mobility,
shadowing and demand do not depend on the actions, so they are generated at
reset; ``rsrp_at`` / ``demand_at`` can therefore be queried for any slot, which
the model-based oracle baselines use for lookahead.
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class EnvConfig:
    # network
    n_cells: int = 7
    isd: float = 200.0            # inter-site distance [m]
    fc_ghz: float = 3.5
    n_prb: int = 100
    w_prb: float = 180e3          # [Hz]
    p_tx_dbm: float = 30.0        # cell Tx power over all PRBs
    pl_exp: float = 3.5
    shadow_std: float = 8.0       # [dB]
    shadow_corr: float = 50.0     # decorrelation distance of the shadowing map [m]
    noise_psd: float = -174.0     # [dBm/Hz]
    noise_fig: float = 7.0        # [dB]
    rsrp_min: float = -110.0      # coverage threshold [dBm]
    se_max: float = 0.0           # spectral-efficiency cap [bit/s/Hz], 0 = none (paper)
    n_ue: int = 100
    mob_std: float = 1.0          # random-walk std per slot [m]
    hotspot_frac: float = 0.0     # fraction of UEs dropped in hotspots
    hotspot_mode: str = "points"  # 'points': random spots, Gaussian spread hotspot_std
                                  # 'cells' : uniform discs of hotspot_radius on outer cells
    n_hotspots: int = 2
    hotspot_std: float = 40.0     # [m], 'points'
    hotspot_radius: float = 90.0  # [m], 'cells'
    hotspot_layout: str = "random"  # 'cells': random | nonadjacent | opposite | adjacent
    hotspot_peaks: str = "spread"   # 'cells': hotspot-cell peaks T/n apart, or 'random'
    # traffic and service
    demand_mbps: float = 2.0
    xi_min: float = 0.2
    T: int = 3600                 # slots per cycle (= episode)
    peak_mode: str = "staggered"  # 'staggered' (paper), 'sync', 'jitter'
    peak_jitter: float = 0.1      # std of per-cell peak offset / T for 'jitter'
    beta: float = 0.8
    v_max: float = 0.05
    w_h: float = 0.1
    # control
    cio_min: float = -6.0
    cio_max: float = 6.0
    cio_step: float = 1.0
    cover: bool = True            # every sleeping cell keeps an active neighbour (10f)
    K: int = 60                   # ES epoch [slots]
    T_dwell: int = 300            # minimum dwell [slots]
    W: int = 10                   # demand history window
    # power model (7)
    p_fix: float = 30.0
    delta_p: float = 4.0
    p_sleep: float = 5.0
    zeta: float = 20.0

    def with_(self, **kw):
        return replace(self, **kw)


# Named scenarios.  EnvConfig() itself keeps the values of Table II.
PRESETS = {
    "paper": {},
    # coverage no longer binding, capacity-limited, uniform UEs (first probe)
    "uniform": dict(rsrp_min=-120.0, demand_mbps=2.0),
    # two standing hotspots on non-adjacent outer cells with peaks half a cycle
    # apart.  All-on meets V_max with margin, so congestion comes from the ES
    # decisions; the MLB matters once cells sleep and the hotspot peaks.
    "hotspot": dict(rsrp_min=-120.0, hotspot_frac=0.7, hotspot_mode="cells", n_hotspots=2,
                    hotspot_radius=90.0, hotspot_layout="nonadjacent", demand_mbps=2.2),
    # same at higher load: All-on with CIO = 0 violates V_max, the CIO restores it
    "hotspot_tight": dict(rsrp_min=-120.0, hotspot_frac=0.7, hotspot_mode="cells",
                          n_hotspots=2, hotspot_radius=90.0, hotspot_layout="nonadjacent",
                          demand_mbps=2.5),
}


def make_config(preset="paper", **overrides):
    return EnvConfig(**dict(PRESETS[preset], **overrides))


def hex_centers(isd):
    ang = np.deg2rad(np.arange(6) * 60.0)
    ring = isd * np.stack([np.cos(ang), np.sin(ang)], -1)
    return np.concatenate([np.zeros((1, 2)), ring], 0)


def neighbor_matrix(centers, isd):
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    return (d > 0) & (d < 1.01 * isd)


def in_cluster(p, centers, isd):
    """True where p (..., 2) lies in the union of the hexagonal cells."""
    ang = np.deg2rad(np.array([0.0, 60.0, 120.0]))
    u = np.stack([np.cos(ang), np.sin(ang)], -1)            # (3, 2)
    rel = p[..., None, :] - centers                          # (..., B, 2)
    proj = np.abs(rel @ u.T)                                 # (..., B, 3)
    return (proj <= isd / 2 + 1e-9).all(-1).any(-1)


def evaluate(cfg, rsrp, d, s, theta, rho_int):
    """Association (2), SINR (3), load (5), delivered rate (6), V (9), energy (7).

    rsrp: (..., U, B) dBm   d: (..., U) bit/s   s, theta, rho_int: (..., B)
    Leading dims broadcast, so candidates x samples can be scored in one call.
    The switching term of (7) is added by the caller.
    """
    B = rsrp.shape[-1]
    active = s > 0
    act_u = active[..., None, :]
    rsrp_a = np.where(act_u, rsrp, -np.inf)
    outage = rsrp_a.max(-1) < cfg.rsrp_min
    serv = np.argmax(rsrp_a + theta[..., None, :], -1)
    g = 10.0 ** (rsrp / 10.0)                                # p_b g_{u,b} [mW]
    w = np.where(active, np.minimum(rho_int, 1.0), 0.0)
    full = np.broadcast_shapes(g.shape, w[..., None, :].shape, rsrp_a.shape,
                               np.shape(theta)[:-1] + (1, B))
    g = np.broadcast_to(g, full)
    w_u = np.broadcast_to(w[..., None, :], full)
    serv = np.broadcast_to(serv, full[:-1])
    d = np.broadcast_to(d, full[:-1])
    sig = np.take_along_axis(g, serv[..., None], -1)[..., 0]
    w_s = np.take_along_axis(w_u, serv[..., None], -1)[..., 0]
    interf = (g * w_u).sum(-1) - sig * w_s
    sinr = sig / (interf + cfg_noise_mw(cfg))
    se = np.log2(1.0 + sinr)
    if cfg.se_max > 0:
        se = np.minimum(se, cfg.se_max)
    n = np.where(outage, 0.0, d / (cfg.w_prb * se))
    onehot = (serv[..., None] == np.arange(B)) & ~outage[..., None]
    rho = (n[..., None] * onehot).sum(-2) / cfg.n_prb
    dem = (d[..., None] * onehot).sum(-2)
    rho_s = np.take_along_axis(rho, serv, -1)
    degraded = ~outage & (rho_s > 1.0 / cfg.beta)
    U = rsrp.shape[-2]
    V = (outage.sum(-1) + degraded.sum(-1)) / U
    p_tx_w = 10.0 ** ((cfg.p_tx_dbm - 30.0) / 10.0)
    E = np.where(active, cfg.p_fix + cfg.delta_p * np.minimum(rho, 1.0) * p_tx_w,
                 cfg.p_sleep).sum(-1)
    return dict(serv=serv, outage=outage, rho=rho, dem=dem, V=V,
                out_frac=outage.mean(-1), E=E)


def cfg_noise_mw(cfg):
    return 10.0 ** ((cfg.noise_psd + 10 * np.log10(cfg.w_prb) + cfg.noise_fig) / 10.0)


def energy_ref(cfg):
    p_tx_w = 10.0 ** ((cfg.p_tx_dbm - 30.0) / 10.0)
    return cfg.n_cells * (cfg.p_fix + cfg.delta_p * p_tx_w)


class RanEnv:
    def __init__(self, cfg: EnvConfig, n_envs=1, seed=0):
        assert cfg.n_cells == 7, "hex layout is implemented for the 7-cell cluster"
        self.cfg = cfg
        self.E = n_envs
        self.B = cfg.n_cells
        self.U = cfg.n_ue
        self.rng = np.random.default_rng(seed)
        self.centers = hex_centers(cfg.isd)
        self.nbr = neighbor_matrix(self.centers, cfg.isd)
        self.theta_grid = np.arange(cfg.cio_min, cfg.cio_max + 1e-9, cfg.cio_step)
        self.e_ref = energy_ref(cfg)
        self.pl0 = 20 * np.log10(4 * np.pi * cfg.fc_ghz * 1e9 / 3e8)
        self.p_prb_dbm = cfg.p_tx_dbm - 10 * np.log10(cfg.n_prb)

    # ---------------------------------------------------------------- exogenous
    def _drop(self, n):
        cfg, out = self.cfg, np.empty((0, 2))
        lim = 1.5 * cfg.isd
        while len(out) < n:
            p = self.rng.uniform(-lim, lim, (4 * n, 2))
            out = np.concatenate([out, p[in_cluster(p, self.centers, cfg.isd)]])
        return out[:n]

    def _pick_hot_cells(self):
        """Outer cells carrying a hotspot, subject to hotspot_layout."""
        cfg, n = self.cfg, self.cfg.n_hotspots
        outer = np.arange(1, self.B)

        def ring_dist(i, j):
            k = abs(int(i) - int(j))
            return min(k, len(outer) - k)

        need = {"random": lambda d: True, "nonadjacent": lambda d: d >= 2,
                "opposite": lambda d: d == 3, "adjacent": lambda d: d == 1}[cfg.hotspot_layout]
        for _ in range(1000):
            hc = self.rng.choice(outer, n, replace=False)
            if all(need(ring_dist(a, b)) for i, a in enumerate(hc) for b in hc[i + 1:]):
                return hc
        raise ValueError(f"no {cfg.hotspot_layout} layout with {n} hotspots")

    def _drop_ues(self):
        cfg, E, U = self.cfg, self.E, self.U
        pos = self._drop(E * U).reshape(E, U, 2)
        n_hot = int(round(cfg.hotspot_frac * U))
        self.hot_cells = np.full((E, cfg.n_hotspots), -1)
        if n_hot > 0 and cfg.hotspot_mode == "cells":
            # standing hotspots centred on cells; each UE inherits the profile of
            # its nearest cell, so a hotspot follows the diurnal profile of its cell
            for e in range(E):
                hc = self._pick_hot_cells()
                self.hot_cells[e] = hc
                cell = hc[np.arange(n_hot) % len(hc)]
                a = self.rng.uniform(0, 2 * np.pi, n_hot)
                r = cfg.hotspot_radius * np.sqrt(self.rng.uniform(0, 1, n_hot))
                p = self.centers[cell] + r[:, None] * np.stack([np.cos(a), np.sin(a)], -1)
                bad = ~in_cluster(p, self.centers, cfg.isd)
                p[bad] = self.centers[cell[bad]]
                pos[e, :n_hot] = p
        elif n_hot > 0:
            for e in range(E):
                hs = self._drop(cfg.n_hotspots)
                idx = self.rng.integers(0, cfg.n_hotspots, n_hot)
                p = hs[idx] + self.rng.normal(0, cfg.hotspot_std, (n_hot, 2))
                bad = ~in_cluster(p, self.centers, cfg.isd)
                p[bad] = hs[idx[bad]]
                pos[e, :n_hot] = p
        return pos

    def _shadow_maps(self):
        cfg = self.cfg
        res = 5.0
        lim = 1.5 * cfg.isd + 20.0
        n = int(np.ceil(2 * lim / res)) + 1
        self.sh_res, self.sh_lim, self.sh_n = res, lim, n
        if cfg.shadow_std <= 0:
            return np.zeros((self.E, self.B, n, n), np.float32)
        white = self.rng.normal(size=(self.E, self.B, n, n))
        f = np.fft.fftfreq(n, d=res)
        kx, ky = np.meshgrid(f, f, indexing="ij")
        sig = cfg.shadow_corr / 2.0                      # Gaussian kernel std
        H = np.exp(-2 * (np.pi * sig) ** 2 * (kx ** 2 + ky ** 2))
        m = np.real(np.fft.ifft2(np.fft.fft2(white) * H))
        m = (m - m.mean((-2, -1), keepdims=True)) / m.std((-2, -1), keepdims=True)
        return (cfg.shadow_std * m).astype(np.float32)

    def _walk(self, pos0):
        cfg = self.cfg
        pos = np.empty((cfg.T + 1,) + pos0.shape, np.float32)
        pos[0] = pos0
        p = pos0.copy()
        for t in range(cfg.T):
            q = p + self.rng.normal(0, cfg.mob_std, p.shape)
            ok = in_cluster(q, self.centers, cfg.isd)
            p = np.where(ok[..., None], q, p)
            pos[t + 1] = p
        return pos

    def rsrp_from_pos(self, pos, env_idx=None):
        """pos (..., E', U, 2) -> RSRP (..., E', U, B) in dBm."""
        cfg = self.cfg
        rel = pos[..., None, :] - self.centers                  # (..., E, U, B, 2)
        dist = np.maximum(np.linalg.norm(rel, axis=-1), 1.0)
        pl = self.pl0 + 10 * cfg.pl_exp * np.log10(dist)
        # bilinear lookup in the per-(env, cell) shadowing maps
        g = (pos + self.sh_lim) / self.sh_res
        g = np.clip(g, 0, self.sh_n - 1.001)
        i0 = np.floor(g).astype(int)
        fr = g - i0
        maps = self.sh_maps if env_idx is None else self.sh_maps[env_idx]
        e_ix = np.arange(maps.shape[0])[:, None, None]
        b_ix = np.arange(self.B)[None, None, :]
        ix, iy = i0[..., 0][..., None], i0[..., 1][..., None]
        fx, fy = fr[..., 0][..., None], fr[..., 1][..., None]
        S = ((1 - fx) * (1 - fy) * maps[e_ix, b_ix, ix, iy]
             + fx * (1 - fy) * maps[e_ix, b_ix, ix + 1, iy]
             + (1 - fx) * fy * maps[e_ix, b_ix, ix, iy + 1]
             + fx * fy * maps[e_ix, b_ix, ix + 1, iy + 1])
        return self.p_prb_dbm - pl + S

    def rsrp_at(self, t, env_idx=None):
        t = np.minimum(np.asarray(t), self.cfg.T - 1)
        pos = self.pos[t] if env_idx is None else self.pos[t][..., env_idx, :, :]
        if env_idx is not None and np.ndim(env_idx) == 0:
            pos = pos[..., None, :, :]
            return self.rsrp_from_pos(pos, [env_idx])[..., 0, :, :]
        return self.rsrp_from_pos(pos, env_idx)

    def xi_at(self, t):
        """Diurnal profile (42) per (env, cell); t scalar or array -> (..., E, B)."""
        cfg = self.cfg
        t = np.asarray(t, float)[..., None, None]
        c = (1 + np.cos(2 * np.pi * (t - self.t_peak) / cfg.T)) / 2
        return cfg.xi_min + (1 - cfg.xi_min) * c

    def demand_at(self, t, env_idx=None):
        xi = self.xi_at(t)                                       # (..., E, B)
        d = self.cfg.demand_mbps * 1e6 * np.take_along_axis(
            xi, np.broadcast_to(self.home, xi.shape[:-2] + self.home.shape), -1)
        return d if env_idx is None else d[..., env_idx, :]

    # --------------------------------------------------------------- dynamics
    def reset(self):
        cfg, E, B = self.cfg, self.E, self.B
        self.sh_maps = self._shadow_maps()
        pos0 = self._drop_ues()
        self.pos = self._walk(pos0)
        d0 = np.linalg.norm(pos0[..., None, :] - self.centers, axis=-1)
        self.home = np.argmin(d0, -1)                            # c(u), (E, U)
        if cfg.peak_mode == "staggered":
            self.t_peak = self.rng.uniform(0, cfg.T, (E, B))
        elif cfg.peak_mode == "sync":
            self.t_peak = np.repeat(self.rng.uniform(0, cfg.T, (E, 1)), B, 1)
        elif cfg.peak_mode == "jitter":
            self.t_peak = (self.rng.uniform(0, cfg.T, (E, 1))
                           + self.rng.normal(0, cfg.peak_jitter * cfg.T, (E, B)))
        else:
            raise ValueError(cfg.peak_mode)
        if (cfg.hotspot_mode == "cells" and cfg.hotspot_frac > 0
                and cfg.hotspot_peaks == "spread"):
            # hotspot cells peak T/n apart, so the congested cell moves over the cycle
            base = self.rng.uniform(0, cfg.T, E)
            n = cfg.n_hotspots
            for j in range(n):
                self.t_peak[np.arange(E), self.hot_cells[:, j]] = (base + j * cfg.T / n) % cfg.T
        self.t = 0
        self.s = np.ones((E, B), int)
        self.theta = np.zeros((E, B))
        self.tau = np.full((E, B), cfg.T_dwell)
        self.last_serv = np.full((E, self.U), -1)
        # interference fixed point for the initial all-on state
        rsrp, d = self.rsrp_at(0), self.demand_at(0)
        rho = np.ones((E, B))
        for _ in range(4):
            rho = evaluate(cfg, rsrp, d, self.s, self.theta, rho)["rho"]
        self.rho = rho
        self.dem_hist = np.zeros((E, B, cfg.W))
        self.dem_hist[..., 0] = evaluate(cfg, rsrp, d, self.s, self.theta, rho)["dem"]
        return self.obs()

    def eligible(self):
        """Cells allowed to change state at the current slot (dwell, Sec. III-A)."""
        return self.tau >= self.cfg.T_dwell - 1

    def covered(self, s):
        """True where every sleeping cell has an active neighbour (s: (..., B))."""
        on = np.asarray(s) > 0
        has_nbr = (on.astype(float) @ self.nbr.T.astype(float)) > 0
        return on.any(-1) & (on | has_nbr).all(-1)

    def _enforce_cover(self, s_new):
        """Withhold deactivations that would leave a sleeping cell uncovered; the cells
        switched off this epoch are examined in increasing order of load."""
        fixed = np.zeros(self.E, int)
        for e in np.where(~self.covered(s_new))[0]:
            off = np.where((self.s[e] > 0) & (s_new[e] == 0))[0]
            base = s_new[e].copy()
            base[off] = 1
            for b in off[np.argsort(self.rho[e, off])]:
                trial = base.copy()
                trial[b] = 0
                if self.covered(trial):
                    base = trial
                else:
                    fixed[e] += 1
            s_new[e] = base
        return s_new, fixed

    def is_epoch_start(self):
        return self.t % self.cfg.K == 0

    def quantize_theta(self, theta):
        cfg = self.cfg
        th = np.clip(theta, cfg.cio_min, cfg.cio_max)
        return cfg.cio_min + np.round((th - cfg.cio_min) / cfg.cio_step) * cfg.cio_step

    def step(self, s_req, theta):
        cfg = self.cfg
        cover_fixed = np.zeros(self.E, int)
        if self.is_epoch_start():
            s_new = np.where(self.eligible(), np.asarray(s_req, int), self.s)
            if cfg.cover:
                s_new, cover_fixed = self._enforce_cover(s_new)
        else:
            s_new = self.s
        switched = s_new != self.s
        self.tau = np.where(switched, 0, np.minimum(self.tau + 1, cfg.T_dwell))
        theta = self.quantize_theta(theta)
        rsrp, d = self.rsrp_at(self.t), self.demand_at(self.t)
        ev = evaluate(cfg, rsrp, d, s_new, theta, self.rho)
        served = ~ev["outage"]
        ho = served & (self.last_serv >= 0) & (ev["serv"] != self.last_serv)
        self.last_serv = np.where(served, ev["serv"], self.last_serv)
        E_t = ev["E"] + cfg.zeta * switched.sum(-1)
        H_t = ho.mean(-1)
        self.s, self.theta, self.rho = s_new, theta, ev["rho"]
        self.dem_hist = np.roll(self.dem_hist, 1, -1)
        self.dem_hist[..., 0] = ev["dem"]
        self.t += 1
        n_serv = np.stack([(served & (ev["serv"] == b)).sum(-1) for b in range(self.B)], -1)
        info = dict(E=E_t, Ebar=E_t / self.e_ref, V=ev["V"], H=H_t,
                    out=ev["out_frac"], rho=ev["rho"], s=s_new.copy(),
                    theta=theta.copy(), switches=switched.sum(-1), n_serv=n_serv,
                    cover_fixed=cover_fixed)
        info["R_M"] = -info["V"] - cfg.w_h * H_t
        done = self.t >= cfg.T
        return self.obs(), info, done

    def obs(self):
        """Global state x_t of (14): loads, s_{t-1}, theta_{t-1}, demand window, tau."""
        return dict(rho=self.rho.copy(), s=self.s.copy(), theta=self.theta.copy(),
                    D=self.dem_hist.copy(), tau=self.tau.copy(), t=self.t)

    def obs_vector(self):
        """Flat, roughly normalized x_t for learning agents, shape (E, dim)."""
        cfg = self.cfg
        D = self.dem_hist / (cfg.demand_mbps * 1e6 * cfg.n_ue / cfg.n_cells)
        return np.concatenate([
            self.rho, self.s, self.theta / max(abs(cfg.cio_min), cfg.cio_max),
            D.reshape(self.E, -1), self.tau / cfg.T_dwell,
            np.repeat([[(self.t % cfg.K) / cfg.K]], self.E, 0)], -1)
