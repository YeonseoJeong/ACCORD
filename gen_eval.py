"""Load generalization: policies trained with N = 100 UEs, evaluated with N = 50 ... 150 UEs.

Evaluation episodes as evaluate.py (RanEnv(cfg, 5, 1000 k), 150 episodes, sampled actions) with
cfg.n_ue = N.  The hotspot share (70 %) and the per-UE demand are unchanged, so the offered load
scales with N.  The demand window D in the observations is normalized as in training (n_ue of the
run's config, 100), so a policy sees the absolute demand.  --obs_norm per_n normalizes D by the
evaluated N instead (D then carries no information on N; only the loads rho do).

    .venv/bin/python gen_eval.py run allon 150                     # All-on (run first: E0 of the others)
    .venv/bin/python gen_eval.py run rule 150                      # also allon_mlbrule
    .venv/bin/python gen_eval.py run learned 150 runs/exp1_ippo_s1 [--obs_norm per_n]
    .venv/bin/python gen_eval.py table                              # -> results/gen_load.{md,csv}
"""
import argparse
import csv
import json
import os
from dataclasses import replace

import numpy as np
import torch

import evaluate
import trainer
from baselines import AllOnES, RuleES, RuleMLB, ZeroMLB, run
from ran_env import EnvConfig

OUT = "results/gen_load"
NS = (50, 75, 100, 125, 150)
REF_RUN = "complexity_1000/ippo_s1"            # environment config of the N = 100 runs
METHODS = [("All-on", "allon"), ("All-on + MLB rule", "allon_mlbrule"), ("Rule-based (ρ_th 0.3)", "rule"),
           ("ES-only learning", "pidA_es_only_i"), ("Learned ES, MLB off", "pidA_es_cio0_i"),
           ("ACCORD", "exp1_ippo"), ("ACCORD, D normalized by N", "exp1_ippo@pern")]


def patch_obs(n_ref):
    """D as in training: dem_hist / (demand * n_ref / n_cells) whatever the evaluated n_ue."""
    base = trainer.obs_parts

    def parts(env):
        p = base(env)
        p["D"] = p["D"] * env.cfg.n_ue / n_ref
        return p
    evaluate.obs_parts = parts


def rollout(cfg, make, episodes, chunk=5):
    acc = {k: [] for k in ("E_sum", "V", "H", "O", "n_on")}
    for k in range(int(np.ceil(episodes / chunk))):
        n = min(chunk, episodes - k * chunk)
        es, mlb = make()
        r = run(cfg, es, mlb, n_envs=n, seed=1000 * k, record=True)
        r["n_on"] = r["trace"]["n_on"].mean(0)
        for key in acc:
            acc[key].extend(np.asarray(r[key], float).tolist())
    return acc


def job(kind, N, run_dir, obs_norm, episodes):
    os.makedirs(OUT, exist_ok=True)
    if kind == "learned":
        conf, m, cfg0, hp, nets = evaluate.load(run_dir)
        if obs_norm == "train":
            patch_obs(cfg0.n_ue)
        rng = np.random.default_rng(0)
        cfg = replace(cfg0, n_ue=N)
        out = {os.path.basename(run_dir.rstrip("/")) + ("@pern" if obs_norm == "per_n" else ""):
               rollout(cfg, lambda: evaluate.policies(m, nets, hp, False, rng), episodes)}
    else:
        cfg = replace(EnvConfig(**json.load(open(REF_RUN + "/config.json"))["cfg"]), n_ue=N)
        makers = {"allon": {"allon": lambda: (AllOnES(), ZeroMLB())},
                  "rule": {"rule": lambda: (RuleES(0.3), RuleMLB(trigger=0.9)),
                           "allon_mlbrule": lambda: (AllOnES(), RuleMLB(trigger=0.9))}}[kind]
        out = {name: rollout(cfg, mk, episodes) for name, mk in makers.items()}
    for name, acc in out.items():
        json.dump(dict(N=N, episodes=episodes, obs_norm=obs_norm, **acc),
                  open(f"{OUT}/{name}_N{N}.json", "w"))
        print(name, N, f"V {np.mean(acc['V']):.4f}  cells {np.mean(acc['n_on']):.2f}", flush=True)


# ------------------------------------------------------------------------------------------- table
def records(name, N, base):
    r = json.load(open(f"{OUT}/{name}_N{N}.json"))
    E, V = np.array(r["E_sum"]), np.array(r["V"])
    return dict(ES=100 * (1 - E / base["E"]), V=V, dV=V - base["V"], n_on=np.array(r["n_on"]),
                H=np.array(r["H"]))


def table():
    from make_tables import with_ci
    rows = []
    for N in NS:
        if not os.path.exists(f"{OUT}/allon_N{N}.json"):
            continue
        a = json.load(open(f"{OUT}/allon_N{N}.json"))
        base = dict(E=np.array(a["E_sum"]), V=np.array(a["V"]))
        for lab, key in METHODS:
            if "_" in key and not key.startswith("allon"):          # learned: seeds 1-5
                names = [f"{key.split('@')[0]}_s{s}" + ("@pern" if "@" in key else "") for s in range(1, 6)]
            else:
                names = [key]
            if not all(os.path.exists(f"{OUT}/{n}_N{N}.json") for n in names):
                continue
            recs = [records(n, N, base) for n in names]
            p, ci = with_ci(recs)
            sd = float(np.std([r["ES"].mean() for r in recs], ddof=1)) if len(recs) > 1 else float("nan")
            rows.append(dict(method=lab, N=N, seeds=len(recs), ES_seed_sd=sd,
                             **{k: p[k] for k in p}, **{f"{k}_lo": ci[k][0] for k in ci},
                             **{f"{k}_hi": ci[k][1] for k in ci}))
    keys = ["method", "N", "seeds", "ES", "ES_lo", "ES_hi", "ES_seed_sd", "V", "V_lo", "V_hi",
            "dV", "dV_lo", "dV_hi", "n_on", "n_on_lo", "n_on_hi", "H"]
    with open("results/gen_load.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["allon_V_above_vmax"])
        allon_v = {r["N"]: r["V"] for r in rows if r["method"] == "All-on"}
        for r in rows:
            w.writerow([r[k] if isinstance(r[k], (str, int)) else f"{r[k]:.5f}" for k in keys]
                       + [int(allon_v.get(r["N"], 0) > 0.05)])
    get = {(r["method"], r["N"]): r for r in rows}
    md = ["# Load generalization: trained with N = 100 UEs, evaluated with N UEs",
          "", "150 evaluation episodes per N (seeds as the main table), sampled actions; learned rows: "
          "seeds 1-5, two-level bootstrap 95% CI; ΔV̄ = V̄ − V̄(All-on) on the same episodes.", ""]
    for title, k, d, ci in (("Energy saving [%]", "ES", 1, True), ("V̄", "V", 3, True),
                            ("ΔV̄ = V̄ − V̄_All-on", "dV", 3, True), ("Active cells (of 7)", "n_on", 2, False),
                            ("ΔV̄ per 1 % saving", "ratio", 4, False)):
        md += [f"### {title}", "", "| Method | " + " | ".join(f"N = {N}" for N in NS) + " |",
               "|---|" + "---|" * len(NS)]
        for lab, _ in METHODS:
            cells = []
            for N in NS:
                r = get.get((lab, N))
                if r is None:
                    cells.append("")
                elif k == "ratio":
                    cells.append(f"{r['dV'] / r['ES']:.4f}" if r["ES"] > 1 else "--")
                else:
                    c = f"{r[k]:.{d}f}"
                    if ci and r["seeds"] > 1:
                        c += f" [{r[k + '_lo']:.{d}f}, {r[k + '_hi']:.{d}f}]"
                    if k == "ES" and r["seeds"] > 1:
                        c += f" (± {r['ES_seed_sd']:.1f})"
                    cells.append(c)
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
        md.append("")
    feas = [N for N in NS if (("All-on", N) in get and get[("All-on", N)]["V"] > 0.05)]
    md.append(f"All-on exceeds V_max = 0.05 at N = {feas}.")
    open("results/gen_load.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "table"])
    ap.add_argument("kind", nargs="?", choices=["allon", "rule", "learned"])
    ap.add_argument("N", nargs="?", type=int)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--obs_norm", choices=["train", "per_n"], default="train")
    ap.add_argument("--episodes", type=int, default=150)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.mode == "table":
        table()
    else:
        job(a.kind, a.N, a.run_dir, a.obs_norm, a.episodes)


if __name__ == "__main__":
    main()
