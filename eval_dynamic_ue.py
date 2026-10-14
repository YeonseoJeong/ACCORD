#!/usr/bin/env python3
"""Zero-shot dynamic-UE evaluation: ACCORD versus non-learning baselines.
Place next to trainer.py, evaluate.py, ran_env.py and baselines.py.
Existing training artifacts are READ-ONLY; results go to <root>/dynamic_ue_rules/.
Example:
 python eval_dynamic_ue_rules.py --root complexity_1000 --seed 1 \
     --populations 50,75,100,75,50 --phase-slots 720 --episodes 10 \
     --traffic diurnal --window 30
Compare ACCORD, Rule-based (RuleES + RuleMLB), All-on, and
AllOnRuleMLB (AllOnES + RuleMLB) by default. No new learning and no environment reset
between population changes.
The extra UE trajectories are pre-generated; only the first N(t) participate
in the association, interference/load, service metric and handover calculations.
At population changes the current load and newest demand observation are refreshed
before ES/MLB choose actions (zero measurement-delay evaluation).
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate import load, policies
from ran_env import RanEnv, evaluate as phy_evaluate, energy_ref
from baselines import AllOnES, RuleES, RuleMLB, ZeroMLB
class DynamicUERanEnv(RanEnv):
    """Seven-cell RanEnv supporting changes in *active* UE population.
    The underlying simulator precomputes max(population) trajectories, but only
    a prefix participates in each step. A stratified prefix order preserves
    approximately the intended hotspot fraction at every population size.
    Crucially cfg.n_ue remains the TRAINING number, retaining the observation
    normalization in trainer.obs_parts() (D/scale).
    """
    def __init__(self, cfg, n_envs, seed, populations, phase_slots,
                 traffic='diurnal', flat_factor=0.6):
        super().__init__(cfg, n_envs=n_envs, seed=seed)
        self.populations = tuple(int(n) for n in populations)
        self.phase_slots = int(phase_slots)
        self.max_ue = max(self.populations)
        self.traffic = traffic
        self.flat_factor = flat_factor
        self.U = self.max_ue  # RanEnv uses self.U for trajectory generation.
        self.n_active = self.populations[0]
    def _drop_ues(self):
        pos = super()._drop_ues()  # (E, max_ue, 2); hotspot UEs are at the front
        nh = int(round(self.cfg.hotspot_frac * self.U))
        if nh in (0, self.U):
            return pos
        # Keep roughly the same hotspot fraction among first 50, 100, 150, ...
        for e in range(self.E):
            hot = self.rng.permutation(nh)
            other = nh + self.rng.permutation(self.U - nh)
            h, o, order = 0, 0, []
            for i in range(1, self.U + 1):
                if int(round(self.cfg.hotspot_frac * i)) > int(round(self.cfg.hotspot_frac * (i - 1))):
                    order.append(int(hot[h])); h += 1
                else:
                    order.append(int(other[o])); o += 1
            pos[e] = pos[e, order].copy()
        return pos
    def demand_at(self, t, env_idx=None):
        if self.traffic == 'flat':
            shape = (self.E, self.U)
            d = np.full(shape, self.cfg.demand_mbps * 1e6 * self.flat_factor)
            return d if env_idx is None else d[env_idx]
        return super().demand_at(t, env_idx)
    def reset(self):
        super().reset()
        self.n_active = self.populations[0]
        # RanEnv.reset() initializes interference for ALL pre-generated UEs.
        # Correct it to reflect only initially active UEs before first policy.
        n = self.n_active
        rsrp, d = self.rsrp_at(0)[:, :n, :], self.demand_at(0)[:, :n]
        rho = np.ones_like(self.rho)
        for _ in range(4):
            rho = phy_evaluate(self.cfg, rsrp, d, self.s, self.theta, rho)['rho']
        self.rho = rho
        self.dem_hist.fill(0.0)
        self.dem_hist[..., 0] = phy_evaluate(self.cfg, rsrp, d, self.s, self.theta, rho)['dem']
        return self.obs()

    def prepare_population(self):
        """Apply this slot's UE population BEFORE controllers observe the state.

        Recompute rho and the latest demand measurement using the PREVIOUS BS
        activation/CIO, but with the newly active UEs and current-slot radio
        conditions. Older demand-history entries and dwell timers are retained.

        This deliberately assumes current UE demand/load is observable at the
        population-change boundary (zero measurement delay). No future information
        or change to the trained observation normalization is used.
        """
        n = self.populations[min(self.t // self.phase_slots, len(self.populations) - 1)]
        if n == self.n_active:
            return

        self.n_active = n
        # Departing UEs lose serving-cell memory; re-arrivals are new sessions.
        self.last_serv[:, n:] = -1
        rsrp = self.rsrp_at(self.t)[:, :n, :]
        demand = self.demand_at(self.t)[:, :n]

        # Same fixed-point iteration as in RanEnv.reset(), under current cells/CIO.
        # The next ES/MLB actions have not yet been applied.
        rho = self.rho.copy()
        for _ in range(4):
            rho = phy_evaluate(self.cfg, rsrp, demand,
                               self.s, self.theta, rho)['rho']
        self.rho = rho
        # Only refresh the newest observation; keep older history intact.
        self.dem_hist[..., 0] = phy_evaluate(
            self.cfg, rsrp, demand, self.s, self.theta, rho)['dem']

    def step(self, s_req, theta):
        cfg = self.cfg
        # Also safe when step() is called without the usual pre-action preparation.
        self.prepare_population()
        n = self.n_active
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
        rsrp = self.rsrp_at(self.t)[:, :n, :]
        demand = self.demand_at(self.t)[:, :n]
        ev = phy_evaluate(cfg, rsrp, demand, s_new, theta, self.rho)
        served = ~ev['outage']
        prev = self.last_serv[:, :n]
        ho = served & (prev >= 0) & (ev['serv'] != prev)
        self.last_serv[:, :n] = np.where(served, ev['serv'], prev)
        self.last_serv[:, n:] = -1  # returning UE is a new arrival, not a handover
        E_t = ev['E'] + cfg.zeta * switched.sum(-1)
        H_t = ho.mean(-1)
        self.s, self.theta, self.rho = s_new, theta, ev['rho']
        self.dem_hist = np.roll(self.dem_hist, 1, -1)
        self.dem_hist[..., 0] = ev['dem']
        self.t += 1
        n_serv = np.stack([(served & (ev['serv'] == b)).sum(-1) for b in range(self.B)], -1)
        own_rho = np.take_along_axis(ev['rho'], ev['serv'], axis=1)
        delivered = np.where(served, demand / np.maximum(own_rho, 1.0), 0.0)
        info = dict(E=E_t, Ebar=E_t / self.e_ref, V=ev['V'], H=H_t,
                    out=ev['out_frac'], rho=ev['rho'], s=s_new.copy(),
                    theta=theta.copy(), switches=switched.sum(-1), n_serv=n_serv,
                    cover_fixed=cover_fixed,
                    throughput_mbps=delivered.sum(-1) / 1e6,
                    active_ues=n)
        info['R_M'] = -info['V'] - cfg.w_h * H_t
        return self.obs(), info, self.t >= cfg.T


def make_env(cfg, populations, slots, episodes, seed, traffic, flat_factor):
    env = DynamicUERanEnv(cfg, n_envs=episodes, seed=seed,
                          populations=populations, phase_slots=slots,
                          traffic=traffic, flat_factor=flat_factor)
    env.reset()
    return env


def evaluate_method(env, controllers):
    es, mlb = controllers
    es.reset(env)
    mlb.reset(env)
    # Mean across independent evaluation episodes at each time slot.
    keys = ('E', 'V', 'H', 'out', 'throughput_mbps')
    data = {k: np.zeros(env.cfg.T, dtype=float) for k in keys}
    data['on'] = np.zeros(env.cfg.T, dtype=float)
    data['switches'] = np.zeros(env.cfg.T, dtype=float)
    for t in range(env.cfg.T):
        # Population arrival/departure is observed before this slot's actions.
        # At t=720/1440/... ES therefore sees N(t), not the preceding N(t-1).
        env.prepare_population()
        if env.is_epoch_start():
            s_req = es.act(env)
            s_next = np.where(env.eligible(), s_req, env.s)
        else:
            s_req = s_next = env.s
        theta = mlb.act(env, s_next)
        _, info, _ = env.step(s_req, theta)
        for key in keys:
            data[key][t] = np.mean(info[key])
        data['on'][t] = info['s'].sum(-1).mean()
        data['switches'][t] = info['switches'].mean()
        es.observe(env, info)
        mlb.observe(env, info)
    return data

def rolling_mean(x, window):
    # Trailing moving average; handle first slots without zero padding.
    a = np.asarray(x, dtype=float)
    c = np.concatenate(([0.0], np.cumsum(a)))
    k = np.arange(len(a))
    lo = np.maximum(k + 1 - window, 0)
    return (c[k + 1] - c[lo]) / (k + 1 - lo)

def rolling_es(E, refE, window):
    return 100.0 * (1.0 - rolling_mean(E, window) /
                    np.maximum(rolling_mean(refE, window), 1e-12))

def resolve_run(root, name, seed):
    choices = {
        'ACCORD': [f'ippo_s{seed}'],
    }[name]
    for r in choices:
        p = root / r
        if (p / 'final.pt').is_file() and (p / 'config.json').is_file():
            return p
    raise FileNotFoundError(f'{name}: missing final.pt/config.json in '
                            + ', '.join(str(root / p) for p in choices))

def save_csv(path, data):
    names = list(data)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(zip(*(data[k] for k in names)))

def plot_results(outdir, populations, slots, trace, names, v_max, window):
    T = len(trace['t'])
    x = trace['t']
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={'height_ratios': [0.7, 1.25, 1.1, 0.85]})
    ax = axes[0]
    ax.step(x, trace['UEs'], where='post', color='black', linewidth=1.9)
    ax.set_ylabel('Active UEs')
    ax.set_ylim(0, max(populations) * 1.22)
    styles = {
        'ACCORD': dict(color='tab:blue', lw=1.7),
        'Rule-based': dict(color='tab:orange', lw=1.55),
        'All-on': dict(color='gray', lw=1.4, ls='--'),
        'AllOnRuleMLB': dict(color='tab:green', lw=1.45, ls='-.'),
    }
    labels = {'AllOnRuleMLB': 'All-on + Rule MLB'}
    for name in names:
        style = styles[name]
        label = labels.get(name, name)
        axes[1].plot(x, trace[f'{name}_V_ma'], label=label, **style)
        axes[2].plot(x, trace[f'{name}_ES_ma'], label=label, **style)
        axes[3].plot(x, trace[f'{name}_active_BS_ma'], label=label, **style)
    axes[1].axhline(v_max, ls=':', color='red', lw=1.2, label=f'Vmax = {v_max:g}')
    axes[1].set_ylabel(f'Service degradation\n({window}-slot avg.)')
    axes[2].axhline(0, ls=':', color='gray', lw=0.7)
    axes[2].set_ylabel(f'Energy saving (%)\n({window}-slot window)')
    axes[3].set_ylabel('Active BSs')
    axes[3].set_ylim(0, max(7.4, float(np.max([trace[f'{name}_active_BS_ma'].max() for name in names])) + .4))
    axes[-1].set_xlabel('Time (slots)')
    for ax in axes:
        for j in range(1, len(populations)):
            ax.axvline(j * slots, ls='--', lw=0.7, alpha=0.35, color='gray')
        ax.grid(alpha=0.2)
        ax.set_xlim(0, T-1)
    axes[1].legend(loc='upper left', ncol=2, fontsize=8)
    axes[2].legend(loc='best', fontsize=8)
    fig.suptitle('Dynamic UE population: ACCORD vs. non-learning baselines', fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / 'dynamic_ue.png', dpi=220)
    fig.savefig(outdir / 'dynamic_ue.pdf')
    plt.close(fig)

def select_rule_rho_off(requested, probe_path, cfg):
    """Reuse the paper's previously swept threshold when its probe file exists.
    Follow reference_table.py: among feasible ('ruleT') settings, maximize
    energy saving; if none is feasible, choose the one with lowest mean V.
    This uses the original fixed-100-UE probe, NOT the dynamic evaluation data.
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
                    group = groups.setdefault((r['fam'], r['par']), {'V': [], 'E_sum': []})
                    group['V'].extend(r['V'])
                    group['E_sum'].extend(r['E_sum'])
            allon = groups.get(('allon', None))
            options = []
            if allon is not None:
                e_ref = np.asarray(allon['E_sum'], dtype=float)
                for (family, threshold), g in groups.items():
                    if family != 'ruleT':
                        continue
                    e = np.asarray(g['E_sum'], dtype=float)
                    if len(e) != len(e_ref):
                        continue
                    options.append((float(threshold), float(np.mean(g['V'])),
                                    float(np.mean(1.0 - e / e_ref))))
            if options:
                v_max = float(d['cfgs'][key].get('v_max', cfg.v_max))
                feasible = [o for o in options if o[1] <= v_max]
                chosen = max(feasible, key=lambda o: o[2]) if feasible else min(options, key=lambda o: o[1])
                return chosen[0], f'previous fixed-UE probe {path}'
        print(f'WARNING: no matching fixed-UE ruleT probe in {path}.', flush=True)
    else:
        print(f'NOTE: prior probe {path} not found.', flush=True)
    print('Using rho_off=0.2 as an illustrative default; pass --rho-off to '
          'use the exact threshold chosen for your paper.', flush=True)
    return 0.2, 'illustrative fallback (not necessarily paper-tuned)'

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='complexity_1000', help='Directory containing ippo_s1/final.pt and config.json')
    ap.add_argument('--seed', type=int, default=1, help='Training checkpoint seed')
    ap.add_argument('--eval-seed', type=int, default=3000, help='Common evaluation-environment seed')
    ap.add_argument('--episodes', type=int, default=5, help='Parallel independent evaluation trajectories')
    ap.add_argument('--populations', default='50,100,150,100,50')
    ap.add_argument('--phase-slots', type=int, default=720)
    ap.add_argument('--window', type=int, default=60, help='Rolling-average smoothing window in slots')
    ap.add_argument('--methods', default='ACCORD,Rule-based,All-on,AllOnRuleMLB',
                    help='Comma-separated subset of ACCORD,Rule-based,All-on,AllOnRuleMLB')
    ap.add_argument('--rho-off', type=float, default=None,
                    help='RuleES deactivation load threshold. Default: paper probe if found, else 0.2')
    ap.add_argument('--rho-on', type=float, default=0.8,
                    help='RuleES reactivation load threshold (default: 0.8)')
    ap.add_argument('--mlb-thr', type=float, default=0.2,
                    help='RuleMLB neighbor-mean CIO threshold (default: 0.2)')
    ap.add_argument('--mlb-trigger', type=float, default=0.9,
                    help='RuleMLB congestion trigger (default: 0.9, dead-zone baseline)')
    ap.add_argument('--rule-probe', default='results/hotspot_nonadj_d2.0_2.2.json',
                    help='Existing fixed-100-UE probe to reuse the paper-selected rho_off')
    ap.add_argument('--traffic', choices=['diurnal', 'flat'], default='diurnal',
                    help='diurnal: original varying demands; flat: isolate effect of N(t)')
    ap.add_argument('--flat-factor', type=float, default=0.6,
                    help='Per-UE constant demand = this fraction of peak demand, if --traffic flat')
    ap.add_argument('--greedy', action='store_true', help='Deterministic mode rather than stochastic policy')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', default=None, help='Default <root>/dynamic_ue_rules')
    args = ap.parse_args()
    populations = tuple(int(s.strip()) for s in args.populations.split(','))
    if not populations or min(populations) < 1 or args.phase_slots < 1 or args.episodes < 1 or args.window < 1:
        ap.error('Population counts, phase-slots, episodes, window must be positive.')
    methods = [s.strip() for s in args.methods.split(',')]
    choices = {'ACCORD', 'Rule-based', 'All-on', 'AllOnRuleMLB'}
    if not methods or len(methods) != len(set(methods)) or set(methods) - choices:
        ap.error('Use a unique subset of ACCORD,Rule-based,All-on,AllOnRuleMLB in --methods')
    if not (0 <= args.rho_on and args.mlb_thr >= 0 and args.mlb_trigger >= 0):
        ap.error('Rule thresholds must be non-negative.')
    if args.rho_off is not None and args.rho_off < 0:
        ap.error('--rho-off must be non-negative.')
    if args.flat_factor <= 0:
        ap.error('--flat-factor must be positive.')
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    outdir = Path(args.out) if args.out else root / 'dynamic_ue_rules'
    outdir.mkdir(parents=True, exist_ok=True)
    T = args.phase_slots * len(populations)
    rd = resolve_run(root, 'ACCORD', args.seed)
    conf, m, reference, hp, nets = load(str(rd))
    if reference.n_cells != 7:
        raise ValueError('This evaluation requires the 7-cell RanEnv')
    if m['es'] != 'learn' or m['mlb'] != 'learn':
        raise ValueError('Expected an ACCORD checkpoint with learned ES and MLB')
    for net_name in ('es_actor', 'mlb_actor'):
        net = getattr(nets, net_name)
        if net is not None:
            net.eval()
    print(f'Loaded ACCORD: {rd} (training UE count={reference.n_ue})', flush=True)
    rho_off, rule_source = select_rule_rho_off(args.rho_off, args.rule_probe, reference)
    print(f'Rule-based: rho_off={rho_off:g} ({rule_source}), '
          f'rho_on={args.rho_on:g}, MLB threshold={args.mlb_thr:g}, '
          f'MLB trigger={args.mlb_trigger:g}', flush=True)
    cfg_eval = reference.with_(T=T)
    # cfg_eval.n_ue deliberately remains the training count (obs normalization).
    def new_env():
        return make_env(cfg_eval, populations, args.phase_slots, args.episodes,
                        args.eval_seed, args.traffic, args.flat_factor)
    print(f'Evaluating N(t)={populations}; {T} slots; {args.episodes} episodes; traffic={args.traffic}', flush=True)
    baseline = evaluate_method(new_env(), (AllOnES(), ZeroMLB()))
    records = {'All-on': baseline}
    with torch.no_grad():
        for name in methods:
            if name == 'All-on':
                continue
            print(f'EVAL {name} ...', flush=True)
            env = new_env()
            if name == 'ACCORD':
                rng = np.random.default_rng(args.eval_seed + 123)
                controllers = policies(m, nets, hp, greedy=args.greedy, rng=rng)
            elif name == 'Rule-based':
                controllers = (RuleES(rho_off=rho_off, rho_on=args.rho_on),
                               RuleMLB(thr=args.mlb_thr, trigger=args.mlb_trigger))
            elif name == 'AllOnRuleMLB':
                controllers = (AllOnES(), RuleMLB(thr=args.mlb_thr, trigger=args.mlb_trigger))
            records[name] = evaluate_method(env, controllers)
    trace = dict(t=np.arange(T), UEs=np.repeat(populations, args.phase_slots),
                 AllOn_V=baseline['V'], AllOn_V_ma=rolling_mean(baseline['V'], args.window),
                 AllOn_E=baseline['E'])
    for name in methods:
        r = records[name]
        trace[f'{name}_V'] = r['V']
        trace[f'{name}_V_ma'] = rolling_mean(r['V'], args.window)
        trace[f'{name}_E'] = r['E']
        trace[f'{name}_ES_ma'] = rolling_es(r['E'], baseline['E'], args.window)
        trace[f'{name}_active_BS'] = r['on']
        trace[f'{name}_active_BS_ma'] = rolling_mean(r['on'], args.window)
        trace[f'{name}_H'] = r['H']
        trace[f'{name}_throughput_Mbps'] = r['throughput_mbps']
        trace[f'{name}_switches'] = r['switches']
    save_csv(outdir / 'dynamic_ue_trace.csv', trace)
    summary_rows = []
    print('\nPhase summary (same exogenous UE trajectories for all methods):')
    for j, n in enumerate(populations):
        sl = slice(j * args.phase_slots, (j + 1) * args.phase_slots)
        refE = baseline['E'][sl].sum()
        row_base = dict(phase=j + 1, UEs=n, slots=args.phase_slots)
        allon = dict(row_base, method='All-on', V=baseline['V'][sl].mean(),
                     ES_pct=0.0, H=baseline['H'][sl].mean(), active_BS=7.0,
                     switches_per_episode=float(baseline['switches'][sl].sum()),
                     throughput_Mbps=baseline['throughput_mbps'][sl].mean())
        summary_rows.append(allon)
        for name in methods:
            if name == 'All-on':
                continue
            r = records[name]
            s = dict(row_base, method=name, V=r['V'][sl].mean(),
                     ES_pct=100 * (1 - r['E'][sl].sum() / refE),
                     H=r['H'][sl].mean(), active_BS=r['on'][sl].mean(),
                     switches_per_episode=float(r['switches'][sl].sum()),
                     throughput_Mbps=r['throughput_mbps'][sl].mean())
            summary_rows.append(s)
            print(f' phase {j+1}: UEs={n:3d} {name:12s} V={s["V"]:.4f} '
                  f'ES={s["ES_pct"]:6.2f}% ON={s["active_BS"]:.2f} ', flush=True)
    save_csv(outdir / 'dynamic_ue_summary.csv',
             {k: [r[k] for r in summary_rows] for k in summary_rows[0]})
    plot_results(outdir, populations, args.phase_slots, trace, methods, cfg_eval.v_max, args.window)
    meta = dict(training_checkpoints={'ACCORD': str(rd)},
                training_UEs=reference.n_ue, eval_UEs=list(populations), phase_slots=args.phase_slots,
                episodes=args.episodes, eval_seed=args.eval_seed, traffic=args.traffic,
                flat_factor=args.flat_factor if args.traffic == 'flat' else None,
                window=args.window, greedy=args.greedy, v_max=cfg_eval.v_max,
                methods=methods,
                rule_based=dict(rho_off=rho_off, rho_off_source=rule_source,
                                rho_on=args.rho_on, mlb_thr=args.mlb_thr,
                                mlb_trigger=args.mlb_trigger),
                note=('Continuous trajectory without resets; arrivals/departures are changes '
                      'in active UE count. A fixed pool of trajectories is pre-generated. '
                      'Population changes are reflected in the current load and newest demand '
                      'observation before actions (zero measurement delay). The service constraint '
                      'is a long-run average; rolling-window V > Vmax does not alone imply violation.'))
    (outdir / 'dynamic_ue_config.json').write_text(json.dumps(meta, indent=2))
    print(f'\nSaved: {outdir}/dynamic_ue.png, .pdf, _trace.csv, _summary.csv, _config.json', flush=True)


if __name__ == '__main__':
    main()
