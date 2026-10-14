"""Evaluate trained w_h sweep checkpoints on held-out seeds.

The three main metrics are:
  - Energy Saving (%) relative to the same-seed All-On/CIO=0 reference
  - Service Degradation V (slot mean)
  - Handover Rate H (slot mean)

Run beside trainer.py, ran_env.py and baselines.py.

The w_h=0.1 reference is loaded from complexity_1000/ippo_s*/ by default.
The w_h=0.5,1,2,5 runs are loaded from runs/wh_sweep/ippo_wh*_s*/.

Quick start (stochastic action sampling by default):
    python eval_wh.py --method ippo --policy sample

All training seeds discovered automatically:
    python eval_wh.py --method ippo --eval-seeds 2000 2001 2002 2003 2004

Full 24-hour evaluation (when one slot represents one second):
    python eval_wh.py --method ippo --slots 86400 --eval-seeds 2000 2001 2002

Output:
    runs/wh_sweep/evaluation/detail.csv
    runs/wh_sweep/evaluation/summary.csv
    runs/wh_sweep/evaluation/wh_metrics.png
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from baselines import RuleMLB
from ran_env import EnvConfig, RanEnv
from trainer import HP, Nets, ObsSpec, obs_parts, sample_es, sample_mlb, theta_from


def wh_tag(wh):
    return f"{wh:g}".replace(".", "p")


def find_training_runs(root, method, wh, train_seeds):
    """Return (seed, checkpoint-directory) pairs for a sweep value."""
    prefix = f"{method}_wh{wh_tag(wh)}_s"
    found = []
    for path in root.glob(prefix + "*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(re.escape(prefix) + r"(\d+)", path.name)
        if match and (train_seeds is None or int(match.group(1)) in train_seeds):
            found.append((int(match.group(1)), path))
    return sorted(found)


def find_reference_runs(root, method, seeds):
    """Find original w_h=0.1 checkpoints in complexity_1000/{method}_sN/."""
    prefix = f"{method}_s"
    found = []
    paths = [root] if (root / "config.json").is_file() else list(root.glob(prefix + "*"))
    for path in paths:
        if not path.is_dir():
            continue
        match = re.fullmatch(re.escape(prefix) + r"(\d+)", path.name)
        # Also permit --reference-root to point directly at ippo_sN.
        if not match:
            continue
        seed = int(match.group(1))
        if seeds is not None and seed not in seeds:
            continue
        if (path / "config.json").is_file() and (path / "final.pt").is_file():
            found.append((seed, path))
        else:
            print(f"[WARN] Skipping incomplete reference run: {path} "
                  "(need config.json and final.pt)")
    return sorted(found)


def actual_training_wh(saved_cfg, cfg_kw):
    """Respect a method-specific w_h override before the environment default."""
    method_wh = saved_cfg.get("spec", {}).get("w_h")
    return float(method_wh if method_wh is not None else EnvConfig(**cfg_kw).w_h)


def environment_signature(cfg_kw):
    """Compare effective environmental settings (not just explicitly saved overrides)."""
    cfg = EnvConfig(**cfg_kw)
    return {k: json.dumps(v, sort_keys=True, default=str)
            for k, v in vars(cfg).items() if k != "w_h"}


def init_env(cfg_kw, eval_seed, slots=None):
    cfg_kwargs = dict(cfg_kw)
    if slots is not None:
        cfg_kwargs["T"] = slots
    cfg = EnvConfig(**cfg_kwargs)
    # No overlap with trainer.py's 10_000_000 + 100_000*train_seed + iteration.
    env = RanEnv(cfg, 1, seed=30_000_000 + eval_seed)
    env.reset()
    return env


def evaluate_all_on(cfg_kw, eval_seed, slots=None):
    """All cells ON and zero CIO, with identical exogenous evaluation seed."""
    env = init_env(cfg_kw, eval_seed, slots)
    energy_total = 0.0
    for _ in range(env.cfg.T):
        s = np.ones_like(env.s, dtype=int)
        theta = np.zeros_like(env.theta)
        _, info, _ = env.step(s, theta)
        energy_total += float(np.asarray(info["Ebar"]).mean())
    return energy_total / env.cfg.T


def load_policy(run_dir, env):
    with (run_dir / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    m = config["spec"]  # Exact architecture/method flags used to train.
    hp_fields = set(HP.__dataclass_fields__)
    hp = HP(**{k: v for k, v in config["hp"].items() if k in hp_fields})
    hp.envs = 1
    nets = Nets(m, env.cfg, hp, env)
    checkpoint = run_dir / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    # This loads a checkpoint created locally by the user's own training code.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    nets.load_state_dict(ckpt["nets"])
    for network in (nets.es_actor, nets.mlb_actor, nets.v_e, nets.v_m, nets.v_c, nets.v_cm):
        if network is not None:
            network.eval()
    return m, hp, nets, config


def evaluate_policy(cfg_kw, run_dir, eval_seed, slots=None, greedy=False):
    env = init_env(cfg_kw, eval_seed, slots)
    m, hp, nets, saved = load_policy(run_dir, env)
    spec = ObsSpec(m, hp, env.nbr)
    rng = np.random.default_rng(40_000_000 + eval_seed)
    rule = RuleMLB(trigger=0.9) if m["mlb"] == "rule_dz" else None
    if rule is not None:
        rule.reset(env)

    sums = dict(Ebar=0.0, V=0.0, H=0.0)
    for t in range(env.cfg.T):
        parts = obs_parts(env)
        if t % env.cfg.K == 0:
            if m["es"] == "learn":
                es_obs = spec.es(parts)
                eligible = env.eligible()
                action, _ = sample_es(nets, es_obs, eligible, env.s, rng, greedy=greedy)
                s_req = action.astype(int)
            else:
                s_req = np.ones_like(env.s, dtype=int)
            s_next = np.where(env.eligible(), s_req, env.s)
        else:
            s_req = env.s
            s_next = env.s

        if m["mlb"] == "learn":
            mlb_obs = spec.mlb(parts, s_next)
            idx, _ = sample_mlb(nets, mlb_obs, rng, greedy=greedy)
            theta = theta_from(nets, idx, env.theta, env.cfg, hp.mlb_action)
        elif m["mlb"] == "rule_dz":
            theta = rule.act(env, s_next)
        else:
            theta = np.zeros_like(env.theta)

        _, info, _ = env.step(s_req, theta)
        for key in sums:
            sums[key] += float(np.asarray(info[key]).mean())

    return {key: total / env.cfg.T for key, total in sums.items()}, env.cfg.v_max


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(detail, wh_values):
    summary = []
    for wh in wh_values:
        subset = [r for r in detail if r["w_h"] == wh]
        if not subset:
            continue
        row = {"w_h": wh, "n_runs": len(subset),
               "n_train_seeds": len(set(r["train_seed"] for r in subset)),
               "n_eval_seeds": len(set(r["eval_seed"] for r in subset))}
        for key in ("ES_pct", "V", "H"):
            values = np.array([r[key] for r in subset], dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row["feasible_frac"] = float(np.mean([r["feasible"] for r in subset]))
        row["v_max"] = subset[0]["v_max"]
        summary.append(row)
    return summary


def save_figure(summary, target):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV files were saved but figure was skipped.")
        return

    x = np.array([r["w_h"] for r in summary])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    specs = [
        ("ES_pct", "Energy Saving (%)", True),
        ("V", "Service Degradation (V)", False),
        ("H", "Handover Rate (H)", False),
    ]
    for ax, (key, ylabel, _) in zip(axes, specs):
        means = [r[f"{key}_mean"] for r in summary]
        stds = [r[f"{key}_std"] for r in summary]
        ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=4, linewidth=1.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel("Handover penalty weight ($w_h$)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if key == "V":
            limits = [r["v_max"] for r in summary]
            if len(set(limits)) == 1:
                ax.axhline(limits[0], linestyle="--", linewidth=1.2,
                           label=f"$V_{{max}}$ = {limits[0]:g}")
                ax.legend()
    fig.suptitle("ACCORD handover-weight sensitivity (held-out evaluation)")
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained w_h sweep checkpoints.")
    parser.add_argument("--method", default="ippo", help="Method used by sweep_wh.py")
    parser.add_argument("--indir", default="runs/wh_sweep", help="Directory with w_h=0.5,1,2 runs")
    parser.add_argument("--reference-root", "--reference_root", default="complexity_1000",
                        help="Directory containing original w_h=0.1 runs named METHOD_sN")
    parser.add_argument("--reference-seeds", "--reference_seeds", nargs="+", type=int, default=None,
                        help="Training seeds for w_h=0.1; default: use all found reference runs")
    parser.add_argument("--outdir", default=None, help="Output directory (default: INDIR/evaluation)")
    parser.add_argument("--wh", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--train-seeds", "--train_seeds", nargs="+", type=int, default=None,
                        help="Sweep train seeds (w_h>=0.5); default: auto-discover all. "
                             "For w_h=0.1 use --reference-seeds")
    parser.add_argument("--eval-seeds", "--eval_seeds", nargs="+", type=int,
                        default=[2000, 2001, 2002, 2003, 2004])
    parser.add_argument("--slots", type=int, default=None,
                        help="Override horizon T (default: checkpoint's training configuration)")
    parser.add_argument("--policy", choices=("greedy", "sample"), default="sample",
                        help="Action selection: sample from the policy (default) or greedy argmax/threshold")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.slots is not None and args.slots <= 0:
        parser.error("--slots must be positive")
    if not args.eval_seeds:
        parser.error("At least one evaluation seed is required")
    torch.set_num_threads(args.threads)

    root = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else root / "evaluation"
    outdir.mkdir(parents=True, exist_ok=True)
    detail = []
    allon_cache = {}
    found = 0
    first_signature = None
    for wh in args.wh:
        is_reference = abs(wh - 0.1) < 1e-8
        search_root = Path(args.reference_root) if is_reference else root
        runs = (find_reference_runs(search_root, args.method, args.reference_seeds)
                if is_reference else find_training_runs(root, args.method, wh, args.train_seeds))
        if not runs:
            print(f"[WARN] No training runs found for w_h={wh:g} in {search_root}")
            continue
        for train_seed, run_dir in runs:
            with (run_dir / "config.json").open(encoding="utf-8") as f:
                cfg_saved = json.load(f)
            if cfg_saved.get("method", args.method) != args.method:
                raise ValueError(f"{run_dir}: checkpoint method is {cfg_saved.get('method')}, "
                                 f"not {args.method}")
            if "seed" in cfg_saved and int(cfg_saved["seed"]) != train_seed:
                raise ValueError(f"{run_dir}: folder seed {train_seed} disagrees with "
                                 f"config seed {cfg_saved['seed']}")
            cfg_kw = cfg_saved["cfg"]
            saved_wh = actual_training_wh(cfg_saved, cfg_kw)
            if abs(saved_wh - wh) > 1e-8:
                raise ValueError(f"{run_dir}: expected w_h={wh:g}, "
                                 f"but checkpoint used w_h={saved_wh:g}")
            signature = environment_signature(cfg_kw)
            if first_signature is None:
                first_signature = signature
            elif signature != first_signature:
                differences = [key for key in (first_signature.keys() | signature.keys())
                               if first_signature.get(key) != signature.get(key)]
                print(f"[WARN] {run_dir} has different environment settings besides w_h: "
                      f"{', '.join(sorted(differences))}. "
                      "These results may not be directly comparable.", flush=True)
            found += 1
            print(f"[LOAD] w_h={wh:g}, training seed={train_seed}: {run_dir}", flush=True)
            for eval_seed in args.eval_seeds:
                cache_cfg = dict(cfg_kw)
                # w_h affects reward, not the physical environment; normalize using
                # identical same-seed All-On reference across the sweep.
                cache_cfg.pop("w_h", None)
                cache_key = (json.dumps(cache_cfg, sort_keys=True), eval_seed, args.slots)
                if cache_key not in allon_cache:
                    allon_cache[cache_key] = evaluate_all_on(cfg_kw, eval_seed, args.slots)
                ref_energy = allon_cache[cache_key]
                values, v_max = evaluate_policy(cfg_kw, run_dir, eval_seed, args.slots,
                                                greedy=(args.policy == "greedy"))
                if ref_energy <= 0:
                    raise ValueError("All-On energy reference must be positive")
                es_pct = 100.0 * (ref_energy - values["Ebar"]) / ref_energy
                row = dict(w_h=wh, train_seed=train_seed, eval_seed=eval_seed,
                           checkpoint_dir=str(run_dir),
                           ES_pct=es_pct, V=values["V"], H=values["H"],
                           Ebar=values["Ebar"], Ebar_allon=ref_energy,
                           v_max=float(v_max), feasible=int(values["V"] <= v_max))
                detail.append(row)
                print(f"w_h={wh:g} train={train_seed} eval={eval_seed} | "
                      f"ES={es_pct:6.2f}%  V={values['V']:.4f}  "
                      f"H={values['H']:.5f}  {'OK' if row['feasible'] else 'VIOL'}",
                      flush=True)
    if found == 0:
        raise SystemExit(f"No checkpoints found under {root} or {args.reference_root}. "
                         "Check --method / --indir / --reference-root / --wh.")

    fields_detail = ["w_h", "train_seed", "eval_seed", "checkpoint_dir", "ES_pct", "V", "H", "Ebar",
                     "Ebar_allon", "v_max", "feasible"]
    save_csv(outdir / "detail.csv", detail, fields_detail)
    summary = aggregate(detail, args.wh)
    fields_summary = ["w_h", "n_runs", "n_train_seeds", "n_eval_seeds",
                      "ES_pct_mean", "ES_pct_std", "V_mean", "V_std", "H_mean", "H_std",
                      "feasible_frac", "v_max"]
    save_csv(outdir / "summary.csv", summary, fields_summary)
    save_figure(summary, outdir / "wh_metrics.png")

    print("\nSummary (mean +/- SD over train-seed x eval-seed runs)")
    print(f"{'w_h':>5} | {'ES (%)':>17} | {'V':>17} | {'H':>19} | {'Feasible':>9}")
    print("-" * 84)
    for r in summary:
        print(f"{r['w_h']:5g} | {r['ES_pct_mean']:7.2f} +/- {r['ES_pct_std']:5.2f} | "
              f"{r['V_mean']:7.4f} +/- {r['V_std']:6.4f} | "
              f"{r['H_mean']:8.5f} +/- {r['H_std']:7.5f} | "
              f"{r['feasible_frac']*100:7.1f}%")
    print(f"\nSaved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
