from argparse import ArgumentParser
from pathlib import Path

from controllers.es_controller import AllOnESController, RuleBasedESController
from controllers.mlb_controller import FixedCIOController, RuleBasedMLBController
from env.env import load_env

import csv
import numpy as np

'''
    Controller와 Environment를 연결하는 Runner
'''
def make_es_observation(obs: dict) -> dict: # env.state -> es_controller.observe()
    return {
        "load": obs["load"].copy(),
        "activation": obs["activation"].copy(),
        "dwell_timer": obs["dwell_timer"].copy(),
    }

def make_mlb_observation(obs: dict) -> dict: # env.state -> mlb_controller.observe(), rule-based에서는 ES, MLB가 서로의 제어 파라미터를 직접 관측 X
    return {
        "load": obs["load"].copy(),
        "cio_db": obs["cio_db"].copy(),
    }

def run(
        policy: str = "all_on", 
        seed: int = 0, 
        config_path: str | Path | None = None,
        save_history: bool = False, 
        output_dir: str | Path = "results"
    ) -> dict:

    if policy not in ("all_on", "rule_based"):
        raise ValueError(f"Unknown policy: {policy}")

    if config_path is None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "env" / "config.yaml"

    env = load_env(config_path) 
    obs, _ = env.reset(seed=seed)

    if policy == "all_on":
        es_controller = AllOnESController(env.B, env.K)
        mlb_controller = FixedCIOController(env.B)
    else:
        es_controller = RuleBasedESController(
            env.cell_xy, env.isd, env.K, env.Tdwell,
            off_threshold=0.2, on_threshold=0.6
        )
        mlb_controller = RuleBasedMLBController(
            env.cell_xy, env.isd,
            cio_min=env.cio_min, cio_max=env.cio_max, cio_step=env.cio_step,
            load_threshold=0.2
        )
    es_controller.reset()
    mlb_controller.reset()

    totals = {
        "service_degradation": 0.0,
        "handover_rate": 0.0,
        "normalized_energy": 0.0,
        "energy_w": 0.0,
        "outage_fraction": 0.0,
        "switching_count": 0.0,
    }

    system_history = [] # network performance metrics 
    cell_history =[]    # BS data

    steps = 0
    for t in range(env.T):
        es_action = es_controller.act(make_es_observation(obs), t) # 60 slots마다 action, 그 외에는 None
        mlb_action = mlb_controller.act(make_mlb_observation(obs), t)
        prev_activation = obs["activation"].copy()
        prev_dwell = obs["dwell_timer"].copy()
        prev_load = obs["load"].copy()
        diagnostic = None
        if policy == "rule_based" and es_action is not None:
            diagnostic = es_controller.last_decision

        obs, rewards, terminated, truncated, info = env.step( # runner -> env
            {"es": es_action, "mlb": mlb_action}
        )
        es_controller.observe(make_es_observation(obs))
        steps += 1
        for key in totals:
            totals[key] += info["metrics"][key]

        if save_history:
            metrics = info["metrics"]

            serving = info["serving_cell"]
            demand = info["demand_bps"]
            delivered = info["delivered_rate_bps"]
            outage = info["outage"]

            system_history.append({
                "t": t,
                "total_demand_mbps": np.sum(demand) / 1e6,
                "total_delivered_mbps": np.sum(delivered) / 1e6,
                "energy_w": metrics["energy_w"],
                "normalized_energy": metrics["normalized_energy"],
                "service_degradation": metrics["service_degradation"],
                "outage_fraction": metrics["outage_fraction"],
                "handover_rate": metrics["handover_rate"],
                "switching_count": metrics["switching_count"],
                "jain_load_fairness": metrics["jain_load_fairness"],
                "Vmax": env.Vmax,
            })

            home_demand = np.bincount(env.home_cell, weights=demand, minlength=env.B)
            valid = (serving >= 0) & (serving < env.B)

            serving_demand = np.bincount(serving[valid], weights=demand[valid], minlength=env.B)
            serving_delivered = np.bincount(serving[valid], weights=delivered[valid], minlength=env.B)

            connected = valid & (~outage)
            connected_count = np.bincount(serving[connected], minlength=env.B)

            for b in range(env.B):
                requested = (int(es_action[b]) if es_action is not None else "")
                eligible = (
                    int(info["es_eligible_mask"][b]) if info["es_epoch_start"] else ""
                )
                raw_action = (
                    int(diagnostic["raw_requested"][b]) if diagnostic is not None else ""
                )
                blocked_dwell = (
                    int(diagnostic["blocked_dwell"][b]) if diagnostic is not None else ""
                )

                cell_history.append({
                    "t": t,
                    "bs": b,
                    "home_demand_mbps": home_demand[b] / 1e6,
                    "serving_demand_mbps": serving_demand[b] / 1e6,
                    "delivered_mbps": serving_delivered[b] / 1e6,
                    "connected_ues": int(connected_count[b]),
                    "load": float(obs["load"][b]),
                    "activation": int(obs["activation"][b]),
                    "cio_db": float(obs["cio_db"][b]),
                    "dwell_before":int(prev_dwell[b]),
                    "dwell_timer":int(obs["dwell_timer"][b]),
                    "load_before":float(prev_load[b]),
                    "load":float(obs["load"][b]),

                    "es_raw_action": raw_action,
                    "es_requested": requested,
                    "es_applied": int(obs["activation"][b]),
                    "es_eligible": eligible,
                    "es_blocked_dwell": blocked_dwell,

                    "switched": int(
                        obs["activation"][b] != prev_activation[b]
                    ),
                    "es_safeguard":(int(diagnostic["safeguard_applied"]) if diagnostic is not None else "")
                })

        if terminated or truncated:
            break

    result = {key: value / steps for key, value in totals.items() if key != "switching_count"} # swtiching_count를 제외하고 나머지는 평균 계산
    result["switching_count"] = int(totals["switching_count"])
    result["slots"] = steps
    result["policy"] = policy
    result["seed"] = seed
    result["service_limit_satisfied"] = result["service_degradation"] <= env.Vmax

    if save_history:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{policy}_seed{seed}"

        def save_csv(path, rows):
            if not rows:
                return
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        save_csv(output_dir / f"{prefix}_system.csv", system_history)
        save_csv(output_dir / f"{prefix}_cell.csv", cell_history)
        save_csv(output_dir / f"{prefix}_summary.csv", [result])
    return result

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--policy", type=str, default="all_on", choices=["all_on", "rule_based"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--save_history", action="store_true")
    parser.add_argument("--output_dir", type=Path, default="results")
    args = parser.parse_args()

    result = run(args.policy, args.seed, args.config, args.save_history, args.output_dir)
    for name, value in result.items():
        print(f"{name:>25s}: {value}")

if __name__ == "__main__":
    main()
