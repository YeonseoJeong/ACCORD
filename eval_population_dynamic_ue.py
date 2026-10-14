#!/usr/bin/env python3
"""Evaluate trained final.pt checkpoints with two DIFFERENT UE-population tests.

(A) population_table.csv: independent fixed-population episodes, 3600 slots each,
    one set of episodes per UE count.  Means across slots and evaluation episodes.
(B) dynamic_population.png: a single continuous 3600-slot experiment in which
    population changes between phases WITHOUT resetting BS states / CIO / dwell.

Both experiments compare EXACTLY FIVE methods:
  1. ACCORD (fixed-100)  : fixed-100 UE trained DTDE ACCORD (ippo)
  2. ACCORD (dynamic)    : population-randomized DTDE ACCORD (ippo)
  3. ES-only (dynamic)   : population-randomized ES-only (learned ES, RuleMLB)
  4. Rule-based          : RuleES + RuleMLB (no training)
  5. All-on              : AllOnES + ZeroMLB (all 7 BSs ON, CIO=0)

Each method is evaluated with the same exogenous UE trajectories (within each
experiment), and energy saving uses All-on as the measured reference.

Dependencies (same directory): ran_env.py, trainer.py, evaluate.py, baselines.py.
No training is performed and original checkpoint files are never modified.
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate import load, policies
from ran_env import RanEnv, evaluate as phy_evaluate
from baselines import AllOnES, RuleES, RuleMLB, ZeroMLB


METHOD_ORDER = ('ACCORD (fixed-100)', 'ACCORD (dynamic)',
                'ES-only (dynamic)', 'Rule-based', 'All-on')


def select_rule_rho_off(requested, probe_path, cfg):
    """Use the same fixed-100 UE rule-ES sweep as eval_dynamic_ue_rules.

    Select best energy saving among feasible thresholds; if none is feasible,
    select the lowest average degradation. Never tune on these evaluation sets.
    """
    if requested is not None:
        return requested, 'user-supplied'
    path = Path(probe_path)
    if path.is_file():
        d = json.loads(path.read_text())
        matches = [name for name, kw in d['cfgs'].items()
                   if np.isclose(float(kw.get('demand_mbps', -1)), cfg.demand_mbps)]
        if len(matches) == 1:
            key = matches[0]
            groups = {}
            for r in d['rows']:
                if r['config'] == key and r['fam'] in ('ruleT', 'allon'):
                    group = groups.setdefault((r['fam'], r['par']),
                                              {'V': [], 'E_sum': []})
                    group['V'].extend(r['V'])
                    group['E_sum'].extend(r['E_sum'])
            allon = groups.get(('allon', None))
            if allon:
                e_ref = np.asarray(allon['E_sum'], dtype=float)
                options = []
                for (fam, threshold), vals in groups.items():
                    if fam != 'ruleT' or len(vals['E_sum']) != len(e_ref):
                        continue
                    E = np.asarray(vals['E_sum'], dtype=float)
                    options.append((float(threshold), float(np.mean(vals['V'])),
                                    float(np.mean(1.0 - E / e_ref))))
                if options:
                    vmax = float(d['cfgs'][key].get('v_max', cfg.v_max))
                    feasible = [x for x in options if x[1] <= vmax]
                    chosen = (max(feasible, key=lambda x: x[2]) if feasible
                              else min(options, key=lambda x: x[1]))
                    return chosen[0], f'fixed-UE rule probe: {path}'
        print(f'WARNING: no matching fixed-UE rule probe at {path}', flush=True)
    else:
        print(f'NOTE: fixed-UE rule probe not found: {path}', flush=True)
    print('WARNING: RuleES rho_off=0.2 fallback (not necessarily paper-tuned); '
          'pass --rho-off to select an explicit threshold.', flush=True)
    return 0.2, 'illustrative fallback'


class DynamicUERanEnv(RanEnv):
    """Fixed UE trajectory pool; only N(t) UEs participate in the PHY and costs."""
    def __init__(self, cfg, n_envs, seed, populations, phase_slots,
                 traffic='diurnal', flat_factor=0.6):
        super().__init__(cfg, n_envs=n_envs, seed=seed)
        self.populations = tuple(int(n) for n in populations)
        self.phase_slots = int(phase_slots)
        self.max_ue = max(self.populations)
        self.U = self.max_ue  # pool size; cfg.n_ue remains observation normalization
        self.n_active = self.populations[0]
        self.traffic = traffic
        self.flat_factor = flat_factor

    def _drop_ues(self):
        # Keep the hotspot fraction approximately constant in every UE prefix.
        pos = super()._drop_ues()
        n_hot = int(round(self.cfg.hotspot_frac * self.U))
        if n_hot in (0, self.U):
            return pos
        for e in range(self.E):
            hot = self.rng.permutation(n_hot)
            ordinary = n_hot + self.rng.permutation(self.U - n_hot)
            h = o = 0
            order = []
            for i in range(1, self.U + 1):
                if round(self.cfg.hotspot_frac * i) > round(self.cfg.hotspot_frac * (i - 1)):
                    order.append(int(hot[h])); h += 1
                else:
                    order.append(int(ordinary[o])); o += 1
            pos[e] = pos[e, order].copy()
        return pos

    def demand_at(self, t, env_idx=None):
        if self.traffic == 'flat':
            d = np.full((self.E, self.U), self.cfg.demand_mbps * 1e6 * self.flat_factor)
            return d if env_idx is None else d[env_idx]
        return super().demand_at(t, env_idx)

    def reset(self):
        super().reset()
        self.n_active = self.populations[0]
        n = self.n_active
        rsrp, d = self.rsrp_at(0)[:, :n, :], self.demand_at(0)[:, :n]
        rho = np.ones_like(self.rho)
        for _ in range(4):
            rho = phy_evaluate(self.cfg, rsrp, d, self.s, self.theta, rho)['rho']
        self.rho = rho
        self.dem_hist.fill(0.0)
        self.dem_hist[..., 0] = phy_evaluate(self.cfg, rsrp, d, self.s,
                                              self.theta, rho)['dem']
        return self.obs()

    def prepare_population(self):
        """Refresh load and current demand BEFORE ES/MLB act on an arrival/departure.

        Assumes new population/load is observable at the change boundary (zero
        measurement delay).  Older demand history and BS/dwell/CIO are retained.
        """
        n = self.populations[min(self.t // self.phase_slots, len(self.populations)-1)]
        if n == self.n_active:
            return
        self.n_active = n
        self.last_serv[:, n:] = -1  # departures / re-arrivals are not handovers
        rsrp = self.rsrp_at(self.t)[:, :n, :]
        demand = self.demand_at(self.t)[:, :n]
        rho = self.rho.copy()
        for _ in range(4):
            rho = phy_evaluate(self.cfg, rsrp, demand, self.s, self.theta, rho)['rho']
        self.rho = rho
        self.dem_hist[..., 0] = phy_evaluate(
            self.cfg, rsrp, demand, self.s, self.theta, rho)['dem']

    def step(self, s_req, theta):
        self.prepare_population()
        cfg, n = self.cfg, self.n_active
        fixed = np.zeros(self.E, dtype=int)
        if self.is_epoch_start():
            s_new = np.where(self.eligible(), np.asarray(s_req, int), self.s)
            if cfg.cover:
                s_new, fixed = self._enforce_cover(s_new)
        else:
            s_new = self.s
        switched = s_new != self.s
        self.tau = np.where(switched, 0, np.minimum(self.tau + 1, cfg.T_dwell))
        theta = self.quantize_theta(theta)
        rsrp = self.rsrp_at(self.t)[:, :n, :]
        d = self.demand_at(self.t)[:, :n]
        ev = phy_evaluate(cfg, rsrp, d, s_new, theta, self.rho)
        served = ~ev['outage']
        prev = self.last_serv[:, :n]
        ho = served & (prev >= 0) & (ev['serv'] != prev)
        self.last_serv[:, :n] = np.where(served, ev['serv'], prev)
        self.last_serv[:, n:] = -1
        energy = ev['E'] + cfg.zeta * switched.sum(-1)
        self.s, self.theta, self.rho = s_new, theta, ev['rho']
        self.dem_hist = np.roll(self.dem_hist, 1, axis=-1)
        self.dem_hist[..., 0] = ev['dem']
        self.t += 1
        n_serv = np.stack([(served & (ev['serv'] == b)).sum(-1)
                           for b in range(self.B)], axis=-1)
        own_rho = np.take_along_axis(ev['rho'], ev['serv'], axis=1)
        delivered = np.where(served, d / np.maximum(own_rho, 1.0), 0.0)
        H = ho.mean(-1)
        info = dict(E=energy, Ebar=energy / self.e_ref, V=ev['V'], H=H,
                    out=ev['out_frac'], rho=ev['rho'], s=s_new.copy(),
                    theta=theta.copy(), switches=switched.sum(-1),
                    n_serv=n_serv, cover_fixed=fixed,
                    throughput_mbps=delivered.sum(-1) / 1e6,
                    active_ues=n)
        info['R_M'] = -info['V'] - cfg.w_h * H
        return self.obs(), info, self.t >= cfg.T


def evaluate_controller(cfg, populations, phase_slots, episodes, seed,
                        traffic, flat_factor, es, mlb):
    env = DynamicUERanEnv(cfg, n_envs=episodes, seed=seed,
                          populations=populations, phase_slots=phase_slots,
                          traffic=traffic, flat_factor=flat_factor)
    env.reset()
    es.reset(env)
    mlb.reset(env)
    # Store one time-series averaged over episodes for the graph and table.
    traces = {k: np.empty(cfg.T, dtype=np.float64) for k in ('E', 'V', 'on')}
    for t in range(cfg.T):
        env.prepare_population()  # UE switch before policy action
        if env.is_epoch_start():
            s_req = es.act(env)
            s_next = np.where(env.eligible(), s_req, env.s)
        else:
            s_req = s_next = env.s
        theta = mlb.act(env, s_next)
        _, info, _ = env.step(s_req, theta)
        traces['E'][t] = np.mean(info['E'])
        traces['V'][t] = np.mean(info['V'])
        traces['on'][t] = np.mean(info['s'].sum(-1))
        es.observe(env, info)
        mlb.observe(env, info)
    return traces


def running_mean(arr, window):
    x = np.asarray(arr, dtype=float)
    cs = np.r_[0, np.cumsum(x)]
    stop = np.arange(1, len(x) + 1)
    start = np.maximum(0, stop - window)
    return (cs[stop] - cs[start]) / (stop - start)


def save_table(path, rows):
    fieldnames = ('Method', 'UEs', 'Avg_Active_BS', 'Service_Degradation',
                  'Energy_Saving_pct')
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_dynamic(path, pops, phase_slots, results, window, v_max):
    T = len(next(iter(results.values()))['E'])
    t = np.arange(T)
    ue_curve = np.repeat(pops, phase_slots)
    ref = results['All-on']
    colors = {'ACCORD (fixed-100)': 'tab:purple',
              'ACCORD (dynamic)': 'tab:blue',
              'ES-only (dynamic)': 'tab:orange',
              'Rule-based': 'tab:green',
              'All-on': 'gray'}
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                            gridspec_kw={'height_ratios': [0.7, 1.2, 1, 1]})
    axs[0].step(t, ue_curve, where='post', color='black', linewidth=1.7)
    axs[0].set_ylabel('Active UEs')
    axs[0].set_ylim(0, max(pops) * 1.18)
    for name in METHOD_ORDER:
        r = results[name]
        c = colors.get(name)
        ls = '--' if name == 'All-on' else '-'
        axs[1].plot(t, running_mean(r['V'], window), label=name, color=c,
                    lw=1.5, ls=ls)
        saving = 100 * (1 - running_mean(r['E'], window) /
                        np.maximum(running_mean(ref['E'], window), 1e-12))
        axs[2].plot(t, saving, label=name, color=c, lw=1.5, ls=ls)
        axs[3].plot(t, running_mean(r['on'], window), label=name, color=c,
                    lw=1.5, ls=ls)
    axs[1].axhline(v_max, color='firebrick', ls=':', lw=1.3,
                   label=f'Vmax = {v_max:g}')
    axs[1].set_ylabel(f'Service degradation\n({window}-slot avg.)')
    axs[2].set_ylabel('Energy saving (%)\nvs. All-on')
    axs[3].set_ylabel('Active BSs')
    axs[3].set_ylim(0, 7.4)
    axs[-1].set_xlabel('Time (slots)')
    for ax in axs:
        for ix in range(1, len(pops)):
            ax.axvline(ix * phase_slots, color='gray', ls=':', alpha=0.5, lw=0.9)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, T-1)
    axs[1].legend(fontsize=8, ncol=2)
    fig.suptitle(f'Dynamic UE population (continuous {T}-slot evaluation)')
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--accord', default='runs_dynamic_ue/ippo_s1',
                    help='Population-randomized ACCORD checkpoint directory')
    ap.add_argument('--es-only', default='runs_dynamic_ue/es_only_s1',
                    help='Random-UE ES-only checkpoint directory')
    ap.add_argument('--fixed-accord', default='complexity_1000/ippo_s1',
                    help='Original fixed-100 UE ACCORD checkpoint directory')
    ap.add_argument('--ckpt', default='final.pt')
    ap.add_argument('--populations', default='50,75,100,125,150')
    ap.add_argument('--slots', type=int, default=3600,
                    help='Independent fixed-pop evaluation horizon and dynamic horizon')
    ap.add_argument('--phase-slots', type=int, default=720,
                    help='Duration of each population phase in dynamic graph')
    ap.add_argument('--episodes', type=int, default=10,
                    help='Number of independent evaluation environments per method')
    ap.add_argument('--eval-seed', type=int, default=3000)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--traffic', choices=['diurnal', 'flat'], default='diurnal')
    ap.add_argument('--flat-factor', type=float, default=0.6)
    ap.add_argument('--rho-off', type=float, default=None,
                    help='RuleES deactivation threshold; auto-select from fixed-UE probe if available')
    ap.add_argument('--rho-on', type=float, default=0.8,
                    help='RuleES reactivation threshold')
    ap.add_argument('--mlb-thr', type=float, default=0.2,
                    help='RuleMLB neighbor-mean CIO threshold')
    ap.add_argument('--mlb-trigger', type=float, default=0.9,
                    help='RuleMLB congestion trigger')
    ap.add_argument('--rule-probe', default='results/hotspot_nonadj_d2.0_2.2.json',
                    help='Fixed-UE rule threshold sweep file (if present)')
    ap.add_argument('--stochastic', action='store_true',
                    help='Sample learned actions (default: deterministic argmax/threshold)')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', default='runs_dynamic_ue/population_eval')
    args = ap.parse_args()
    pops = tuple(int(x.strip()) for x in args.populations.split(','))
    if not pops or min(pops) < 1 or args.episodes < 1 or args.window < 1:
        ap.error('Populations, episodes, and smoothing window must be positive')
    if len(pops) * args.phase_slots != args.slots:
        ap.error('Need len(populations) * phase-slots == slots (default 5*720=3600)')
    if args.flat_factor <= 0:
        ap.error('--flat-factor must be positive')
    if (args.rho_off is not None and args.rho_off < 0) or min(
            args.rho_on, args.mlb_thr, args.mlb_trigger) < 0:
        ap.error('Rule-based load thresholds must be nonnegative')
    torch.set_num_threads(args.threads)

    names_dirs = {
        'ACCORD (fixed-100)': args.fixed_accord,
        'ACCORD (dynamic)': args.accord,
        'ES-only (dynamic)': args.es_only,
    }
    models = {}
    for name, run_dir in names_dirs.items():
        p = Path(run_dir)
        if not (p / 'config.json').is_file() or not (p / args.ckpt).is_file():
            ap.error(f'{name}: expected config.json and {args.ckpt} in {p}')
        conf, m, cfg, hp, nets = load(str(p), ckpt=args.ckpt)
        if m['es'] != 'learn' or (name.startswith('ACCORD') and
                                (conf['method'] != 'ippo' or m['mlb'] != 'learn')):
            ap.error(f'{name}: unexpected saved algorithm {conf["method"]}')
        if name.startswith('ES-only') and m['mlb'] != 'rule_dz':
            ap.error(f'{name}: checkpoint should be es_only, got {conf["method"]}')
        for net in (nets.es_actor, nets.mlb_actor):
            if net is not None:
                net.eval()
        models[name] = (m, cfg.with_(T=args.slots), hp, nets)
        print(f'Loaded {name}: {p / args.ckpt}; obs UE ref={cfg.n_ue}', flush=True)

    base_cfg = models['ACCORD (dynamic)'][1]
    # Differences in observation scaling cfg.n_ue are okay; physical model
    # and UE trajectory RNG parameters must otherwise match for fair comparison.
    base_dict = asdict(base_cfg)
    for name, (_, cfg, _, _) in models.items():
        diffs = [k for k, v in base_dict.items()
                 if k != 'n_ue' and asdict(cfg)[k] != v]
        if diffs:
            ap.error(f'{name}: environment parameters differ from ACCORD: {diffs}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    max_ue = max(pops)
    rngseed = args.eval_seed + 123
    rho_off, rule_source = select_rule_rho_off(args.rho_off, args.rule_probe, base_cfg)
    print('Rule-based: RuleES(rho_off={}, rho_on={}); '
          'RuleMLB(thr={}, trigger={}); source={}'.format(
              rho_off, args.rho_on, args.mlb_thr, args.mlb_trigger,
              rule_source), flush=True)

    def run(name, phase_pops, phase_slots):
        if name == 'All-on':
            cfg = base_cfg
            es, mlb = AllOnES(), ZeroMLB()
        elif name == 'Rule-based':
            cfg = base_cfg
            es, mlb = (RuleES(rho_off=rho_off, rho_on=args.rho_on),
                       RuleMLB(thr=args.mlb_thr, trigger=args.mlb_trigger))
        else:
            m, cfg, hp, nets = models[name]
            rng = np.random.default_rng(rngseed)
            es, mlb = policies(m, nets, hp, greedy=not args.stochastic, rng=rng)
        return evaluate_controller(cfg, phase_pops, phase_slots,
                                   args.episodes, args.eval_seed,
                                   args.traffic, args.flat_factor, es, mlb)

    rows = []
    print('\n[1/2] Fixed population: each N independently evaluated for',
          f'{args.slots} slots', flush=True)
    with torch.no_grad():
        for n in pops:
            # A never-reached second phase maintains the SAME 150-UE pool
            # across all fixed-N tests; population stays n for all slots.
            fixed_pops = (n, max_ue)
            baseline = run('All-on', fixed_pops, args.slots)
            print(f'  N={n:3d}: All-on V={baseline["V"].mean():.5f};',
                  end=' ', flush=True)
            for name in METHOD_ORDER:
                data = baseline if name == 'All-on' else run(
                    name, fixed_pops, args.slots)
                row = {'Method': name, 'UEs': n,
                       'Avg_Active_BS': f'{data["on"].mean():.4f}',
                       'Service_Degradation': f'{data["V"].mean():.5f}',
                       'Energy_Saving_pct':
                           f'{100 * (1 - data["E"].sum() / baseline["E"].sum()):.3f}'}
                rows.append(row)
                print(f'{name}: ON {row["Avg_Active_BS"]}, '
                      f'V {row["Service_Degradation"]}, '
                      f'ES {row["Energy_Saving_pct"]}%;', end=' ', flush=True)
            print(flush=True)

        save_table(out / 'population_table.csv', rows)
        print('\n[2/2] Dynamic population, ONE continuous', args.slots,
              '-slot trajectory:', pops, flush=True)
        traces = {}
        for name in METHOD_ORDER:
            print(f'  Evaluating {name}', flush=True)
            traces[name] = run(name, pops, args.phase_slots)
    plot_dynamic(out / 'dynamic_population.png', pops, args.phase_slots,
                 traces, args.window, base_cfg.v_max)
    print(f'\nSaved table:  {out / "population_table.csv"}')
    print(f'Saved figure: {out / "dynamic_population.png"}')


if __name__ == '__main__':
    main()
