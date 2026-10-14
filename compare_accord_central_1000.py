"""Reproducible 1000-iteration ACCORD (ippo) vs centralized PPO benchmark.
Place this file alongside trainer.py, evaluate.py, ran_env.py, baselines.py, probe.py.
Requires torch, numpy, psutil. Does not modify the training implementation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.path.insert(0, os.getcwd())

import numpy as np

# METHODS = {'ACCORD': 'ippo', 'Centralized': 'central'}
METHODS = {"CTDE": "mappo"}


def get_rows(path):
    with open(path, newline='') as f:
        return [{k: float(v) for k, v in r.items() if v not in ('', None)} for r in csv.DictReader(f)]


def first_feasible(vals, steps, limit, sustained=3):
    """First point at which `sustained` consecutive checkpoints are feasible."""
    for i in range(len(vals) - sustained + 1):
        if all(np.isfinite(v) and v <= limit for v in vals[i:i+sustained]):
            return int(steps[i])
    return None


def timed_training(cmd, logpath, interval=0.15, append=False):
    """Wall time and peak aggregate process-tree RSS (sampling-based)."""
    import psutil
    start = time.perf_counter()
    peak = 0
    with open(logpath, 'a' if append else 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                procs = [proc] + proc.children(recursive=True)
                rss = sum(x.memory_info().rss for x in procs if x.is_running())
                peak = max(peak, rss)
            except psutil.Error:
                pass
            time.sleep(interval)
        code = p.wait()
    if code:
        raise RuntimeError(f'Training failed ({code}), see {logpath}')
    return time.perf_counter()-start, peak / 2**20


def inference(run_dir, repeats=1000, warmup=100, threads=2):
    """Batch=1 end-to-end software policy sampling (no environment step/network I/O)."""
    import torch
    from evaluate import load
    from trainer import ObsSpec, obs_parts, sample_es, sample_mlb
    from ran_env import RanEnv
    torch.set_num_threads(threads)
    _, m, cfg, hp, nets = load(str(run_dir))
    env = RanEnv(cfg, 1, 2026)
    env.reset()
    spec = ObsSpec(m, hp, nets.nbr)
    parts = obs_parts(env)
    oe = spec.es(parts)
    # Unlike the actual central rollout, this benchmarks MLB with s_next already available.
    om = spec.mlb(parts, env.s)
    elig = env.eligible()
    rng = np.random.default_rng(5678)
    count = lambda net: sum(p.numel() for p in net.parameters()) if net is not None else 0

    def bench(fn):
        for _ in range(warmup):
            fn()
        times = np.empty(repeats)
        for i in range(repeats):
            t = time.perf_counter_ns()
            fn()
            times[i] = (time.perf_counter_ns()-t)/1e6
        return float(times.mean()), float(np.percentile(times, 95))

    es_mean, es_p95 = bench(lambda: sample_es(nets, oe, elig, env.s, rng))
    mlb_mean, mlb_p95 = bench(lambda: sample_mlb(nets, om, rng))
    total_params = sum(count(getattr(nets, k)) for k in
                       ('es_actor', 'mlb_actor', 'v_e', 'v_m', 'v_c', 'v_cm'))
    return dict(es_ms=es_mean, es_p95_ms=es_p95, mlb_ms=mlb_mean,
                mlb_p95_ms=mlb_p95, actor_params=count(nets.es_actor)+count(nets.mlb_actor),
                train_params=total_params)


def evaluate_checkpoints(run_dir, steps, n_episodes, threads):
    """Held-out stochastic evaluation, same evaluation seeds for both methods."""
    import torch
    from evaluate import evaluate
    torch.set_num_threads(threads)
    out = []
    for it in steps:
        ckpt = f'ckpt_{it}.pt' if (run_dir/f'ckpt_{it}.pt').exists() else 'final.pt'
        r = evaluate(str(run_dir), episodes=n_episodes, chunk=min(5, n_episodes),
                     greedy=False, ckpt=ckpt)
        out.append(dict(iter=it, V=r['V'], ES=r['ES'], H=r['H'], ok=r['ok'], n=r['n']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--checkpoint-every', type=int, default=50)
    ap.add_argument('--eval-episodes', type=int, default=10,
                    help='Held-out episodes per checkpoint; 30 recommended for publication')
    ap.add_argument('--stable-checkpoints', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--envs', type=int, default=16)
    ap.add_argument('--out', default='complexity_1000')
    ap.add_argument('--reuse', action='store_true', help='Reuse completed training runs')
    ap.add_argument('--resume', action='store_true', help='Resume interrupted training and skip completed runs')
    ap.add_argument('--skip-eval', action='store_true')
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    all_rows, curves = [], []
    # Preserve previously completed methods, including ACCORD and Centralized.
    if True:
        previous = root/'per_seed.csv'
        if previous.exists():
            with open(previous, newline='') as f:
                all_rows = list(csv.DictReader(f))
            for row in all_rows:
                for k, v in list(row.items()):
                    if k not in ('method', 'seed'):
                        row[k] = None if v in ('', 'None') else float(v)
                row['seed'] = int(row['seed'])
        previous = root/'checkpoint_evaluations.csv'
        if previous.exists():
            with open(previous, newline='') as f:
                curves = list(csv.DictReader(f))
    for method_name, key in METHODS.items():
        for seed in args.seeds:
            run_dir = root / f'{key}_s{seed}'
            run_dir.mkdir(parents=True, exist_ok=True)
            info_path = run_dir/'timing.json'
            if (args.reuse or args.resume) and (run_dir/'final.pt').exists():
                timing = json.loads(info_path.read_text()) if info_path.exists() else {}
            else:
                cmd = [sys.executable, 'trainer.py', '--method', key, '--preset', 'hotspot',
                       '--iters', str(args.iters), '--envs', str(args.envs), '--seed', str(seed),
                       '--threads', str(args.threads), '--save_every', str(args.checkpoint_every),
                       '--out', str(run_dir)]
                if args.resume:
                    cmd.append('--resume')
                print(f'TRAIN {method_name} seed={seed} ...', flush=True)
                elapsed, rss = timed_training(cmd, run_dir/'train.stdout.log', append=args.resume)
                old_timing = json.loads(info_path.read_text()) if args.resume and info_path.exists() else {}
                old_wall = old_timing.get('wall_s')
                if old_wall is None and args.resume and (run_dir/'log.csv').exists():
                    old_logs = get_rows(run_dir/'log.csv')
                    # Already resumed: log 'sec' is elapsed across sessions, not wall-time measurement.
                    old_wall = max(0.0, old_logs[-1]['sec'] - elapsed) if old_logs else 0.0
                timing = {'wall_s': (old_wall or 0.0) + elapsed,
                          'peak_rss_mb': max(rss, old_timing.get('peak_rss_mb', 0.0))}
                info_path.write_text(json.dumps(timing, indent=2))
            logs = get_rows(run_dir/'log.csv')
            if len(logs) != args.iters:
                raise ValueError(f'{run_dir}: expected {args.iters} log rows, got {len(logs)}')
            from ran_env import EnvConfig
            v_max = EnvConfig(**json.loads((run_dir/'config.json').read_text())['cfg']).v_max
            steps_train = [int(x['iter']) for x in logs]
            train_V = [x['V'] for x in logs]
            # Raw train V can transiently cross threshold; report sustained window too.
            train_first = first_feasible(train_V, steps_train, v_max, 1)
            train_sustained = first_feasible(train_V, steps_train, v_max, 20)
            latency = inference(run_dir, repeats=args.repeats, threads=args.threads)
            if not args.skip_eval:
                steps = list(range(args.checkpoint_every, args.iters+1, args.checkpoint_every))
                if steps[-1] != args.iters:
                    steps.append(args.iters)
                print(f'EVAL  {method_name} seed={seed}: {len(steps)} checkpoints', flush=True)
                results = evaluate_checkpoints(run_dir, steps, args.eval_episodes, args.threads)
                curves = [c for c in curves if not (c['method'] == method_name and int(c['seed']) == seed)]
                for x in results:
                    curves.append(dict(method=method_name, seed=seed, **x))
                eval_first = first_feasible([x['V'] for x in results], steps, v_max,
                                            args.stable_checkpoints)
            else:
                eval_first = None
            record = dict(method=method_name, seed=seed,
                          time_per_iter_s=timing.get('wall_s', float('nan'))/args.iters,
                          peak_rss_mb=timing.get('peak_rss_mb', float('nan')),
                          v_max=v_max,
                          train_first_feasible_iter=train_first,
                          train_20_consecutive_iter=train_sustained,
                          eval_stable_feasible_iter=eval_first,
                          **latency)
            all_rows = [r for r in all_rows if not (r['method'] == method_name and int(r['seed']) == seed)]
            all_rows.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            write_csv(root/'per_seed.csv', all_rows)
            if curves:
                write_csv(root/'checkpoint_evaluations.csv', curves)
    keys = ['time_per_iter_s', 'peak_rss_mb', 'es_ms', 'es_p95_ms',
            'mlb_ms', 'mlb_p95_ms', 'actor_params', 'train_params',
            'train_first_feasible_iter', 'train_20_consecutive_iter',
            'eval_stable_feasible_iter']
    with open(root/'summary.txt', 'w') as f:
        for name in dict.fromkeys(r['method'] for r in all_rows):
            rr = [r for r in all_rows if r['method'] == name]
            print(f'\n{name} (n={len(rr)} seeds)', file=f)
            for k in keys:
                vals = [r[k] for r in rr if r[k] is not None and np.isfinite(r[k])]
                if vals:
                    print(f'  {k}: {statistics.mean(vals):.4f} ± '
                          f'{statistics.stdev(vals) if len(vals)>1 else 0:.4f} '
                          f'(observed {len(vals)}/{len(rr)})', file=f)
                else:
                    print(f'  {k}: N/A (not observed)', file=f)
    print(f'\nResults saved to {root}/summary.txt and CSV files.')


def write_csv(path, rows):
    keys = list(rows[0])
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    temp.replace(path)


if __name__ == '__main__':
    main()
