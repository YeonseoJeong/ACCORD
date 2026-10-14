"""Episode-level randomized active UE population for the original 7-BS RAN.

Use with trainer_dynamic_ue.py.  The original ran_env.py is NOT modified.

Each of n_envs samples N_e ~ DiscreteUniform[ue_min, ue_max] independently
at reset.  Exactly N_e UEs are active for that environment's whole episode;
all N_e generate demand, move, associate, and count in V / handovers.

The simulator generates max_ue trajectories and masks UE indices >= N_e.
This vectorizes differing population sizes without changing actor input shapes.
The first N_e are stratified so the hotspot fraction remains approximately
constant across population sizes (same convention as eval_dynamic_ue_rules.py).

Important: cfg.n_ue is the fixed OBSERVATION-normalization reference (100),
not the active UE count.  env.U == max_ue is the allocated trajectory pool.
"""

import numpy as np

from ran_env import RanEnv, evaluate as phy_evaluate


class RandomPopulationRanEnv(RanEnv):
    """Original PHY and control dynamics, with independently sampled UE counts."""

    def __init__(self, cfg, n_envs=16, seed=0, ue_min=50, ue_max=150):
        if not (1 <= ue_min <= ue_max):
            raise ValueError('Require 1 <= ue_min <= ue_max')
        super().__init__(cfg, n_envs=n_envs, seed=seed)
        self.ue_min = int(ue_min)
        self.ue_max = int(ue_max)
        self.U = self.ue_max  # Fixed tensor dimension, NOT active UE count.
        self.n_active = np.full(self.E, self.ue_max, dtype=np.int64)
        self.active_mask = np.ones((self.E, self.U), dtype=bool)

    def _drop_ues(self):
        """Interleave hotspot / ordinary UEs for a representative prefix."""
        pos = super()._drop_ues()
        n_hot = int(round(self.cfg.hotspot_frac * self.U))
        if n_hot == 0 or n_hot == self.U:
            return pos
        for e in range(self.E):
            hot = self.rng.permutation(n_hot)
            ordinary = n_hot + self.rng.permutation(self.U - n_hot)
            order = []
            h = o = 0
            for i in range(1, self.U + 1):
                # Approximately round(hotspot_frac * i) hotspots in first i UEs.
                target = int(round(self.cfg.hotspot_frac * i))
                previous = int(round(self.cfg.hotspot_frac * (i - 1)))
                if target > previous:
                    order.append(hot[h])
                    h += 1
                else:
                    order.append(ordinary[o])
                    o += 1
            pos[e] = pos[e, order].copy()
        return pos

    def reset(self):
        # Sample once per environment / rollout, then hold fixed for all T slots.
        self.n_active = self.rng.integers(
            self.ue_min, self.ue_max + 1, size=self.E, dtype=np.int64
        )
        self.active_mask = np.arange(self.U)[None, :] < self.n_active[:, None]
        # RanEnv.reset() calls self.demand_at(): our override already masks
        # inactive UEs during initial load/interference fixed-point updates.
        return super().reset()

    def demand_at(self, t, env_idx=None):
        """Real UE demand for all active UEs; zero for inactive trajectories."""
        d = super().demand_at(t, env_idx)
        mask = self.active_mask if env_idx is None else self.active_mask[env_idx]
        return np.where(mask, d, 0.0)

    def step(self, s_req, theta):
        """Same ES/MLB and radio model as RanEnv; active-only UE metrics."""
        cfg = self.cfg
        cover_fixed = np.zeros(self.E, dtype=int)
        if self.is_epoch_start():
            s_new = np.where(self.eligible(), np.asarray(s_req, int), self.s)
            if cfg.cover:
                s_new, cover_fixed = self._enforce_cover(s_new)
        else:
            s_new = self.s
        switched = s_new != self.s
        self.tau = np.where(switched, 0, np.minimum(self.tau + 1, cfg.T_dwell))
        theta = self.quantize_theta(theta)

        rsrp = self.rsrp_at(self.t)  # (n_envs, max_ue, B)
        demand = self.demand_at(self.t)  # inactive UEs have zero demand
        ev = phy_evaluate(cfg, rsrp, demand, s_new, theta, self.rho)

        active = self.active_mask
        served = active & ~ev['outage']
        ho = served & (self.last_serv >= 0) & (ev['serv'] != self.last_serv)
        self.last_serv = np.where(served, ev['serv'], self.last_serv)
        self.last_serv[~active] = -1

        # The original evaluate() divides by max_ue.  Replace its *reported*
        # service metrics with per-environment active-UE denominators.
        own_rho = np.take_along_axis(ev['rho'], ev['serv'], axis=-1)
        degraded = (~ev['outage']) & (own_rho > 1.0 / cfg.beta)
        V = ((ev['outage'] | degraded) & active).sum(axis=-1) / self.n_active
        outage = (ev['outage'] & active).sum(axis=-1) / self.n_active
        H = ho.sum(axis=-1) / self.n_active

        energy = ev['E'] + cfg.zeta * switched.sum(-1)
        self.s, self.theta, self.rho = s_new, theta, ev['rho']
        self.dem_hist = np.roll(self.dem_hist, 1, axis=-1)
        self.dem_hist[..., 0] = ev['dem']
        self.t += 1
        n_serv = np.stack([
            (served & (ev['serv'] == b)).sum(-1) for b in range(self.B)
        ], axis=-1)

        info = dict(E=energy, Ebar=energy / self.e_ref, V=V, H=H,
                    out=outage, rho=ev['rho'], s=s_new.copy(), theta=theta.copy(),
                    switches=switched.sum(-1), n_serv=n_serv,
                    cover_fixed=cover_fixed, active_ues=self.n_active.copy())
        info['R_M'] = -V - cfg.w_h * H
        return self.obs(), info, self.t >= cfg.T
