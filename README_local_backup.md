# ACCORD: environment and non-learning probes

**To reproduce the experiments, follow [EXPERIMENTS.md](EXPERIMENTS.md).** This README
describes the files and keeps the development log.

**Naming (since 2026-10-05).** The proposed ACCORD is the DTDE structure, method key `ippo`
(actors and critics on each agent's own observation). Method key `accord` is the ablation row
ACCORD (CTDE). The Status sections below are the development log and use the older names:
there "IPPO" is today's ACCORD, "ACCORD" is ACCORD (CTDE), and most runs used 500 iterations.
Results for the paper use 1000 iterations and the PID multiplier (kp 20, ki 0.1), which is
the trainer default; `--lam_ctrl i` gives the integral update (38). Paper runs: `runs/pidA_*`
(ACCORD = `pidA_ippo`), integral-update ablation `runs/exp1_ippo_*` (EXPERIMENTS.md).

Simulator of *Learning to Coexist* (Sec. II, V-A) and the non-learning controllers
used to choose a setting in which a learned ES/MLB pair has something to win
before ACCORD itself is trained.

## Files

| file | contents |
|---|---|
| `ran_env.py` | vectorized environment (`RanEnv`, `EnvConfig`), the evaluator of (2)–(9), named scenarios `PRESETS` |
| `baselines.py` | All-on, rule-based ES/MLB, `CioOracleMLB`, `OracleES`, episode runner `run` |
| `probe.py` | runs every baseline family over many seeds and reports the best feasible energy saving per family |
| `plot_probe.py` | energy/degradation frontiers and heatmaps from a probe result |
| `trace_plot.py` | one-episode traces (active cells, demand profiles, hotspot load) |
| `reference_table.py` | reference table at the main setting (All-on, rule-based, oracle) for comparing learned methods |
| `trainer.py` | PPO trainer for ACCORD, the learned baselines and the ablations |
| `evaluate.py` | scores trained runs on the 30 evaluation episodes |
| `run_batch.sh` | trains and evaluates methods × seeds in parallel (`PREFIX`, `PRESET`, `BASE`, `ITERS`, `JOBS`) |
| `collect_results.py` | gathers the evaluations into the results table (mean ± std over seeds) |
| `plot_training.py` | training curves (λ, Ĉ, active cells, share of θ = 0) |
| `coverage_check.py`, `gradient_check.py` | diagnostics: coverage gaps; reward slope of a CIO step |
| `results/` | probe outputs (`*.json`, `*.log`, figures); `results/archive/` holds the first uniform-UE study |

The probes need only `numpy` and `matplotlib`; training needs `torch` (installed in `.venv`).

## Scenarios (`ran_env.PRESETS`)

`EnvConfig()` keeps the values of Table II. The presets change only what is listed.

| preset | changes | purpose |
|---|---|---|
| `paper` | none | Table II as written |
| `uniform` | `rsrp_min=-120`, `demand_mbps=2.0` | coverage no longer binding; capacity-limited, uniform UEs |
| `hotspot` | `rsrp_min=-120`, 70 % of UEs in two standing hotspots (radius 90 m) on non-adjacent outer cells, peaks half a cycle apart, `demand_mbps=2.2` | main setting: All-on meets V_max with margin, the MLB matters once cells sleep |
| `hotspot_tight` | as `hotspot`, `demand_mbps=2.5` | All-on with CIO = 0 violates V_max; the CIO restores it |

`hotspot_layout` places the hotspot cells: `random`, `nonadjacent`, `opposite`, `adjacent`.
Adjacent hotspots make episodes much harder, so the presets exclude them.

## Calibration (hotspot, non-adjacent, 30 seeds, V_max = 0.05)

Energy saving at the best parameter with mean V̄ ≤ V_max; "infeas." = no parameter meets V_max.

`or*` are privileged-information references, not true upper bounds. Each is a heuristic that
sees the future channels and demand. Its numbers come from actually running it, so a better
planner could do better, and a learned policy may exceed them. `orR` stands in for
ES-only learning and `orC` for ACCORD.

The CIO oracle sees only s_{t−1}, as in the paper's simultaneous moves (`sees='prev'`).

| d̄ [Mbps] | V̄ All-on | V̄ All-on + MLB rule | V̄ All-on + CIO oracle | rule | ruleT | ruleES0 | ruleC | or0 | orR | orC |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 0.005 | 0.019 | 0.002 | 3.0 % | 16.1 % | 21.1 % | 18.0 % | 33.1 % | 25.8 % | 33.6 % |
| 2.2 | 0.016 | 0.038 | 0.007 | 0 % | 10.5 % | 13.4 % | 14.9 % | 29.2 % | 18.1 % | 30.9 % |
| 2.3 | 0.026 | 0.050 | 0.012 | −2.6 % | 5.3 % | 11.7 % | 13.1 % | 24.2 % | infeas. | 29.3 % |
| 2.35 | 0.032 | 0.057 | 0.016 | 0 % | 4.6 % | 5.8 % | 12.4 % | 23.2 % | infeas. | 25.0 % |
| 2.5 | 0.054 | 0.081 | 0.028 | infeas. | infeas. | infeas. | 7.3 % | 11.8 % | infeas. | 21.4 % |

- **Measure the MLB's share (`orC − or0`) at matched V̄, not at the best feasible points.**
  - The best feasible points depend on which margin lands just under V_max. Read that way,
    the share jumps around: 0.5, 1.7, 5.1, 1.8 and 9.6 pp for 2.0–2.5 Mbps.
  - Interpolated on each frontier at V̄ = 0.04, it is 2.3, 2.7 and 2.6 pp for 2.2, 2.3 and
    2.35 Mbps.
  - At the same ES plan, the CIO oracle mainly lowers V̄. At margin 0 this is 0.035 → 0.029
    at 2.35 Mbps and 0.057 → 0.040 at 2.5 Mbps. Near capacity this decides whether V_max
    can be met at all.
- **2.2 Mbps is the main setting.** Above it, `orR` can no longer meet V_max and All-on's margin
  shrinks, while the MLB's share at matched V̄ does not grow.
- **The paper's MLB rule raises V̄ even with every cell on.** That is why `orR` is borderline
  at 2.2 Mbps (V̄ = 0.050, 19/30 episodes ok) and infeasible at 2.5 Mbps.

Raw results (`results/`):

| results | file |
|---|---|
| `rule`, `ruleT`, `ruleES0` | `hotspot_nonadj_d2.0_2.2.json`, `hotspot_nonadj_d2.5.json` |
| every family at 2.3 and 2.35 Mbps (`sees='prev'`) | `hotspot_d2.3_2.35.json` |
| `or0`, `orR`, `orC`, run with the CIO oracle seeing s_t | `hotspot_oracle_v2.json` |
| `ruleC` and `orC` with `sees='prev'` | `simultaneous_prev.json` |

## Sensitivity of the CIO oracle (d̄ = 2.2, 30 seeds)

From `results/sens_*.json`:

- **Excess weight `w`** ∈ {0.02, 0.05, 0.1}: no material effect.
  - All-on V̄ moves by ≤ 0.0005.
  - `orC` saving stays at 30.8–30.9 %.
- **Load margin** ∈ {0.10, 0.15, 0.20}: a smaller margin gives a lower V̄ at the same saving.
  - All-on V̄ is 0.004–0.006, 0.006–0.007 and 0.008–0.009 respectively.
  - `orC` at −0.1 has V̄ = 0.032, 0.041 and 0.053.
  - The default 0.15 is therefore conservative.
- **Re-optimisation period**: every slot does slightly better than every 5 slots.
  - All-on V̄ 0.0059 vs 0.0072, with H 0.0135 vs 0.0128.
  - `orC` has the same saving and V̄ 0.036 vs 0.041.
- **Simultaneous (`prev`) vs sequential (`next`) knowledge of s_t**: no material effect.
  - `orC` saving is identical and V̄ differs by ≤ 0.0003.
  - `ruleC` saving differs by ≤ 0.4 pp and V̄ by ≤ 0.005, in both directions.
  - A deactivation leaves the CIO stale for a single slot, at most once per cell per T_dwell.
- **Lookahead covering the full dwell** (adds the T_dwell − 1 sample) changed the oracle results
  by up to ±2 pp. This is the precision to expect when comparing oracle numbers.

## Reference table for the ACCORD experiments

`python reference_table.py` writes `results/reference_d2.2.{md,tex}`. The table lists All-on,
the rule-based pair (dead-zone MLB, best ρ_th) and the oracle reference at the main setting,
with empty rows for the learned methods.

Evaluate learned policies **on the same 30 episodes**: chunk k = 0..5 is
`RanEnv(make_config('hotspot'), n_envs=5, seed=1000 * k)`. `baselines.run` accepts any ES/MLB
object with `reset`, `observe` and `act`, so a trained actor can be wrapped and scored with
the same metrics. Those metrics are energy saving against All-on on the same episode, V̄, Ō,
H̄, Ξ_T, the share of deactivations reversed within T_dwell + K, and Jain's index over active
cells.

## Training the learned methods

`trainer.py` trains ACCORD and the learned baselines with one two-time-scale PPO loop;
`evaluate.py` scores trained runs on the same 30 evaluation episodes as the reference table.
Training uses environment seeds from 10,000,000 upward, so it never sees the evaluation
episodes. The project venv `.venv` holds torch.

```bash
.venv/bin/python trainer.py --method accord --iters 500 --seed 0 --threads 4
.venv/bin/python evaluate.py runs/accord_s0              # greedy (mode) actions
.venv/bin/python evaluate.py runs/accord_s0 --stochastic
```

Methods: `accord`, `es_only`, `ippo`, `team`, `central`, the ablations `accord_sym`,
`accord_no_theta_es`, `accord_no_tau_mlb`, `accord_wh0`, and the sanity checks `mlb_allon`
(all cells on, MLB learns) and `es_cio0` (ES learns, CIO = 0). `--mlb_action delta` replaces
the absolute CIO level of (25) by a −δ/0/+δ step.

Departures from Table II found in the seed-1 checks (`runs/check_v1`, `runs/check2_*`):

- **Dual step.** With η_λ = 0.01, λ stayed below 0.15 while ES switched every cell off within
  about 40 iterations. Once off, V = 1 regardless of small changes, so nothing pulls the
  policy back. The defaults are now η_λ = 0.5 and λ₀ = 1, and Ĉ starts from the first
  rollout.
- **Entropy bonus.** The bonus was summed over cells, which pushed the MLB policy toward
  uniform. It is now the mean over cells.
- **Initialization.** Initial policies start near all-on and θ = 0 (output biases); greedy
  untrained actors reproduce All-on.

### Status (seed 1, 500 iterations, greedy, same 30 episodes, d̄ = 2.2)

| method | saving | V̄ | ok | H̄ | Ξ_T | short | θ = 0 on active cells |
|---|---|---|---|---|---|---|---|
| All-on | 0 % | 0.016 | 27/30 | 0.012 | 0 | -- | -- |
| Rule-based (dead zone) | 10.5 % | 0.049 | 15/30 | 0.040 | 22.0 | 79 % | 55 % |
| `es_only` (learned ES + dead-zone rule MLB) | 13.8 % | 0.044 | 18/30 | 0.040 | 14.7 | 44 % | 55 % |
| `es_cio0` (learned ES, MLB off) | 21.6 % | 0.053 | 17/30 | 0.010 | 12.1 | 34 % | 100 % |
| `accord` | 21.2 % | 0.046 | 19/30 | 0.011 | 10.1 | 17 % | 99 % |
| `mlb_allon` | 0 % | 0.016 | 27/30 | 0.013 | 0 | -- | 98 % |
| Oracle reference | 29.0 % | 0.022 | 28/30 | 0.011 | 20.0 | 27 % | -- |

- **ACCORD's λ levels off** at about 8 from iteration ~350, with Ĉ ≈ V_max. `es_only` and
  `es_cio0` were still creeping up at 500 (Ĉ ≈ 0.06). The greedy–stochastic gap in V shrank
  from 0.035 to 0.002 during training.
- **The learned MLB stays at θ = 0.** It does so with w_h = 0.1 and with w_h = 0. With w_h = 0
  it starts to move late in training, but V gets worse. The cause is a sparse V signal, not
  the handover penalty.
- **ACCORD beats the rule-based pair and ES-only but only matches the learned ES with the MLB
  off.** At 2.2 Mbps the MLB has little to add: about 2 pp at matched V̄ in the oracle probe.
**ES entropy decay at d̄ = 2.2** (`--ent_es_end 0`, `runs/decay_*`). Ĉ now converges below
V_max, at a small cost in saving:

| | no decay | decay |
|---|---|---|
| ACCORD | 21.2 %, V̄ 0.046 (19/30) | 20.1 %, V̄ 0.032 (23/30) |
| learned ES, MLB off | 21.6 %, V̄ 0.053 (17/30) | 18.9 %, V̄ 0.038 (21/30) |

**d̄ = 2.5 (`hotspot_tight`, `runs/t25_*`, no decay)**, compared with the references All-on
V̄ 0.054, All-on + CIO oracle 0.028 and oracle reference 21.4 % at V̄ 0.040:

| method | saving | V̄ | ok | θ ≠ 0 congested / calm | θ hotspot busy | θ others while hotspot busy | V̄ with the MLB switched off |
|---|---|---|---|---|---|---|---|
| `mlb_allon` | 0 % | 0.055 | 17/30 | 5.7 / 0.0 % | +0.06 | −0.01 | -- |
| `es_cio0` | 9.9 % | 0.081 | 12/30 | -- | -- | -- | -- |
| `es_only` (dead-zone rule) | 8.7 % | 0.100 | 8/30 | 85.5 / 25.2 % | −0.26 | +2.26 | -- |
| `accord` | 13.0 % | 0.096 | 8/30 | 7.1 / 0.1 % | −0.16 | +0.03 | 0.099 |
| `accord --mlb_action delta` | 11.6 % | 0.079 | 14/30 | 78.7 / 66.3 % | −0.69 | +0.67 | 0.103 |

- **No learned method met V_max at 2.5 within 500 iterations.** λ reached 12–20. Even with the
  MLB off, the ES did not learn to keep enough cells on (V̄ 0.081, worse than All-on).
- **With the absolute CIO action (25) the MLB stays idle** (96–97 % θ = 0). Switching it off
  changes V̄ by only 0.003.
- **With the incremental action, the MLB works.** It lowers the offset of a congested hotspot
  (−0.69 dB) and raises its neighbours' (+0.67 dB). Switching it off at test time raises V̄
  from 0.079 to 0.103, about as much as the CIO oracle gains over CIO = 0 at All-on (0.026).
  Its offsets persist rather than being switched per slot, so the θ ≠ 0 share is high in calm
  slots too.
- A static check (`gradient_check.py`) shows that a −1 dB step on a congested cell lowers V̄ on
  average (by 0.017 at 2.2 and 0.043 at 2.5). That is far more than its one-off handover cost
  (0.002). The reward has a slope in expectation; the absolute action did not exploit it.
**Incremental CIO action + ES entropy decay** (`--mlb_action delta --ent_es_end 0`,
`runs/dd22_*`, `runs/dd25_*`, `runs/dv02_*`; seed 1, 500 iterations, greedy):

| setting | method | saving | V̄ | ok | V̄ with the MLB switched off | θ hotspot busy / others |
|---|---|---|---|---|---|---|
| 2.2 | `accord` | **24.3 %** | 0.042 | 21/30 | 0.112 | −2.81 / −1.74 dB |
| 2.2 | `es_cio0` | 18.9 % | 0.038 | 21/30 | -- | -- |
| 2.2 | `es_only` | 14.4 % | 0.049 | 18/30 | -- | −0.26 / +1.59 dB |
| 2.2 | `mlb_allon` | 0 % | 0.015 | 27/30 | -- | +0.09 / +0.14 dB |
| 2.5 | `accord` | 14.9 % | 0.091 | 11/30 | 0.136 | −0.17 / +1.40 dB |
| 2.5 | `es_cio0` | 9.3 % | 0.073 | 14/30 | -- | -- |
| 2.2, V_max 0.02 | `accord` | 22.4 % | 0.050 | 6/30 | 0.084 | −1.33 / −0.68 dB |
| 2.2, V_max 0.02 | `es_cio0` | 15.5 % | 0.034 | 18/30 | -- | -- |

- **At the main setting the MLB now does real work.** ACCORD saves 24.3 % at V̄ 0.042, against
  18.9 % for the learned ES with the MLB off, 14.4 % for ES-only and 10.5 % for the
  rule-based pair. That is 84 % of the oracle reference (29.0 %).
- **The ES relies on the MLB.** With the MLB switched off at test time, V̄ rises from 0.042 to
  0.112. The MLB keeps congested hotspots about 1 dB below their neighbours.
- **At 2.5 and at V_max = 0.02, no learned method met the constraint within 500 iterations**
  (λ was still rising). The MLB still lowers V̄ there: by 0.044 at 2.5 and by 0.034 at
  V_max = 0.02.
- **The coverage constraint was never touched** by any of these policies
  (`results/coverage_dd.txt`).
- **Only the CIO offset relative to the neighbours matters for association.** A common shift
  of all offsets changes nothing, so "others" can be negative while hotspots are more negative.

### Main batch (2026-10-05): evaluate with sampled actions

Settings: d̄ = 2.2 Mbps, incremental CIO, ES entropy decay, coverage enforced, 500 iterations,
the same 30 evaluation episodes, sampled (stochastic) execution as in Algorithm 2.
Built with `collect_results.py --override accord=mainmd ippo=mainmd team=mainmd central=mainmd`.

| method | seeds | saving | V̄ | seeds with V̄ ≤ V_max | V̄ greedy | V̄ with the MLB off |
|---|---|---|---|---|---|---|
| ES-only | 5 | 14.8 ± 1.6 % | 0.056 | 2/5 | 0.042 | -- |
| learned ES, MLB off | 5 | 19.2 ± 1.6 % | 0.044 | 4/5 | 0.040 | -- |
| IPPO | 3 | 21.5 ± 1.0 % | 0.035 | 3/3 | 0.055 | 0.081 |
| Team-reward MAPPO-L | 3 | 21.9 ± 1.0 % | 0.040 | 3/3 | 0.053 | 0.080 |
| Centralized | 3 | 21.1 ± 1.1 % | 0.037 | 3/3 | 0.054 | 0.067 |
| ACCORD | 5 | 21.6 ± 2.3 % | 0.035 | 5/5 | 0.062 | 0.077 |

- **Report sampled execution.** The Lagrangian training yields a randomised policy that meets
  the constraint in expectation; optimal CMDP policies may need randomisation (Altman, 1999).
  Taking the mode instead raises V̄ by 0.013–0.027 for every method with a learned MLB and
  breaks the constraint.
- **Decaying the MLB entropy does not close that gap** and does not change the result. ACCORD
  without the MLB decay gives 21.0 ± 1.7 % at V̄ 0.035 (5/5); with a 0.002 floor it gives
  22.4 % (3 seeds). None of the runs locked the MLB at θ = 0.
- **ACCORD beats the learned ES with the MLB off** (+2.4 pp at a lower V̄, all seeds
  feasible) **and ES-only**, which violates V_max on average. Switching off ACCORD's MLB raises
  V̄ to 0.077. Against IPPO, Team-reward and Centralized the difference is within seed noise
  at this setting (3 seeds each).
- **All joint learners leave slack** (V̄ 0.035–0.040 < 0.05, λ still falling), so longer
  training may raise their savings.
- **V_max = 0.03** (`vmax3*`, sampled): ACCORD 20.4 % at V̄ 0.032 (2/3 seeds feasible);
  learned ES with the MLB off 19.2 % at 0.043 (0/3); ES-only 14.1 % at 0.053 (0/3). The
  baselines' λ was still rising at 500 iterations. References (`results/probe_vmax3.json`):
  dead-zone rule pair 3.1 %, oracle ES with CIO = 0 26.3 %, oracle reference 29.0 %.

## Baseline families (`probe.py`)

| family | ES | MLB | role |
|---|---|---|---|
| `allon` | all on | CIO = 0 | energy reference |
| `allon_rule` / `allon_ruleT` | all on | MLB rule / with 0.9 dead zone | what a rule MLB does alone |
| `allon_cio` | all on | `CioOracleMLB` | what the CIO alone can do |
| `rule` / `ruleT` | ES rule | MLB rule / with dead zone | rule-based pair of Sec. V-B |
| `ruleES0` | ES rule | CIO = 0 | |
| `ruleC` | ES rule | `CioOracleMLB` | simple ES on top of an ideal MLB |
| `or0` / `orR` / `orC` | `OracleES` | CIO = 0 / MLB rule / `CioOracleMLB` | `orR` ≈ ES-only learning, `orC` ≈ joint bound |

Summary columns: `headroom = ES*(orC) − best rule-based`, `es_gain = ES*(orC) − ES*(ruleC)`,
`joint = ES*(orC) − ES*(orR)`. `ES*` is the best saving with mean V̄ ≤ V_max; the
`episodes ok` count reports how many episodes meet V_max individually.

## Running

```bash
python probe.py --preset hotspot --sweep demand_mbps=2.2,2.4,2.6 --fams allon_rule,allon_cio
python probe.py --preset hotspot --base hotspot_layout=nonadjacent,demand_mbps=2.5 --seeds 30
python plot_probe.py results/<name>.json
python trace_plot.py --preset hotspot --base hotspot_layout=nonadjacent,demand_mbps=2.5
```

## Things to know

- **The oracles use information no deployed application has.** `OracleES` sees the future
  channels and demand. `CioOracleMLB` optimises the CIO on the true per-UE channels.
  They are heuristics, not optimal, so a learned policy may beat them. The bound on
  decentralized methods is the paper's Centralized baseline.
- **Coverage constraint (10f).** Every sleeping cell must keep an active neighbour. The
  environment enforces it (`EnvConfig.cover`, default on). A deactivation that would break it
  is not executed; the cells switched off in an epoch are examined in increasing order of
  load, and `info["cover_fixed"]` counts the withheld ones. The candidates of `OracleES` satisfy
  it as well.
  - In the 30 evaluation episodes, no learned policy (2.2 and 2.5) and not the oracle reference
    ever left a cell uncovered (`coverage_check.py`, `results/coverage_*.txt`), so their
    results hold under the constraint.
  - RuleES gave up 8 of 354 deactivations for coverage. Without the guard it saves 10.7 %
    instead of 10.5 % but violates V_max (V̄ 0.056).
- **V is cliff-like in the hotspot scenario.** A hotspot cell holds ~40 % of the UEs, and V
  counts all of them once its load passes 1/β = 1.25. The load also moves by about ±0.1 from
  slot to slot. The oracles therefore plan against `1/β − margin`.
- **Episode-to-episode spread is large.** It comes from the UE drop, the shadowing and the
  hotspot placement; adjacent hotspots are much harder. Use ≥ 30 seeds and report how many
  episodes are feasible.
- **Choices not specified in the paper:**
  - path loss is referenced to free space at 1 m
  - shadowing is a spatially correlated map (50 m)
  - UEs are dropped in, and walk inside, the union of the hexagons
  - RuleES uses the epoch maximum, neighbour-triggered reactivation and a coverage guard
  - RuleMLB moves each cell's CIO against the mean load of its active neighbours
