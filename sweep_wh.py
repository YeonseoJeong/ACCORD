"""Independent w_h sweep using the existing trainer.py without editing it.

Place this file beside trainer.py, probe.py, ran_env.py and baselines.py.

Examples:
  python sweep_wh.py --method ippo --iters 500 --seeds 0
  python sweep_wh.py --method ippo --iters 1000 --seeds 0 1 2
  python sweep_wh.py --method accord --iters 500 --seeds 0 --base 'n_ue=100'
"""

import argparse
from pathlib import Path

import torch

from probe import build_configs
from trainer import HP, METHODS, train


def main():
    parser = argparse.ArgumentParser(description="Train separate ACCORD policies at each w_h.")
    parser.add_argument("--method", choices=sorted(METHODS), default="ippo")
    parser.add_argument("--preset", default="hotspot")
    parser.add_argument("--base", default="", help="Environment overrides, e.g. n_ue=100")
    parser.add_argument("--wh", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--mlb_action", choices=("delta", "abs"), default="delta")
    parser.add_argument("--mlb_arch", choices=("flat", "shared"), default="flat")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--outdir", default="runs/wh_sweep")
    args = parser.parse_args()

    if any(w < 0 for w in args.wh):
        parser.error("w_h must be non-negative")
    if args.iters <= 0 or args.envs <= 0:
        parser.error("--iters and --envs must be positive")

    torch.set_num_threads(args.threads)
    original_cfg = next(iter(build_configs(args.preset, args.base, []).values()))

    total = len(args.wh) * len(args.seeds)
    idx = 0
    for wh in args.wh:
        for seed in args.seeds:
            idx += 1
            cfg_kw = dict(original_cfg)
            cfg_kw["w_h"] = wh   # Passed to EnvConfig(**cfg_kw); used by rollout() in MLB reward

            hp = HP(envs=args.envs, mlb_action=args.mlb_action, mlb_arch=args.mlb_arch)
            wh_tag = f"{wh:g}".replace(".", "p")
            output = Path(args.outdir) / f"{args.method}_wh{wh_tag}_s{seed}"
            print(f"\n[{idx}/{total}] START method={args.method} w_h={wh:g} "
                  f"seed={seed} output={output}", flush=True)

            train(
                method=args.method,
                cfg_kw=cfg_kw,
                hp=hp,
                iters=args.iters,
                seed=seed,
                out=str(output),
                save_every=args.save_every,
                log_every=args.log_every,
            )
            print(f"[{idx}/{total}] DONE w_h={wh:g} seed={seed} "
                  f"(config.json, log.csv, final.pt in {output})", flush=True)


if __name__ == "__main__":
    main()
