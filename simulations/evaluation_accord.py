"""
Evaluate a trained ACCORD policy.

Run from repository root, for example:

    python -m simulations.evaluation_accord \
        --checkpoint results/checkpoints/accord_seed0_iter200.pt \
        --eval_seeds 100 101 102 103 104 105 106 107 108 109

Notes
-----
- ES acts every K slots.
- MLB acts every slot.
- Evaluation actions are deterministic.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import torch

from ca_mappo.trainer import CAMAPPOTrainer
from ca_mappo.utils import flatten_normalized_observation, set_seed
from env.env import load_env


def make_rl_state(obs: dict, env) -> np.ndarray:
    return flatten_normalized_observation(
        obs,
        cio_abs_max=max(abs(env.cio_min), abs(env.cio_max)),
        dwell_time=env.Tdwell,
        nominal_demand_bps=env.nominal_demand,
        num_ues=env.U,
    )


# ---------------------------------------------------------------------
# Small utility functions
# ---------------------------------------------------------------------
def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    try:
        return np.asarray(x)
    except Exception:
        return None


def _scalar(x, default=np.nan) -> float:
    if x is None:
        return float(default)

    arr = _to_numpy(x)
    if arr is None or arr.size == 0:
        return float(default)

    return float(np.asarray(arr, dtype=np.float64).mean())


def _find_value(*containers, keys: tuple[str, ...]):
    """
    Find the first matching key from one of the supplied dict-like containers.

    It also searches one level inside common nested dictionaries:
        info["metrics"], info["history"], info["radio"], info["traffic"]
    """
    nested_names = ("metrics", "history", "radio", "traffic", "network", "ue", "bs")

    for container in containers:
        if not isinstance(container, dict):
            continue

        for key in keys:
            if key in container:
                return container[key]

        for nested_name in nested_names:
            nested = container.get(nested_name)
            if not isinstance(nested, dict):
                continue
            for key in keys:
                if key in nested:
                    return nested[key]

    return None


def _vector(x, expected_len: int | None = None):
    if x is None:
        return None

    arr = _to_numpy(x)
    if arr is None:
        return None

    arr = np.asarray(arr).reshape(-1)

    if expected_len is not None and arr.size != expected_len:
        return None

    return arr


def _safe_nanmean(values, default=np.nan):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float(default)
    return float(np.nanmean(arr))


def _safe_nanstd(values, default=np.nan):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float(default)
    return float(np.nanstd(arr))


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------
def load_checkpoint_into_trainer(
    trainer: CAMAPPOTrainer,
    checkpoint_path: str | Path,
) -> dict:
    """
    Prefer CAMAPPOTrainer.load_checkpoint() if your trainer implements it.

    Otherwise fall back to loading common state-dict names.
    If your checkpoint format differs, only this function should need editing.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Best case: trainer already provides its own loader.
    if hasattr(trainer, "load_checkpoint"):
        result = trainer.load_checkpoint(checkpoint_path)
        return result if isinstance(result, dict) else {}

    ckpt = torch.load(
        checkpoint_path,
        map_location=trainer.device,
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise TypeError(
            "Checkpoint is not a dictionary and CAMAPPOTrainer has no "
            "load_checkpoint() method."
        )

    # Candidate attribute/key pairs for common implementations.
    candidates = [
        ("es_actor", ("es_actor", "es_actor_state_dict")),
        ("es_critic", ("es_critic", "es_critic_state_dict")),
        ("mlb_actor", ("mlb_actor", "mlb_actor_state_dict")),
        ("mlb_critic", ("mlb_critic", "mlb_critic_state_dict")),
        ("es_cost_critic", ("es_cost_critic", "cost_critic", "cost_critic_state_dict")),
        ("mlb_cost_critic", ("mlb_cost_critic", "cost_critic", "cost_critic_state_dict")),
    ]

    loaded_any = False

    for attr_name, key_candidates in candidates:
        module = getattr(trainer, attr_name, None)
        if module is None or not hasattr(module, "load_state_dict"):
            continue

        for key in key_candidates:
            if key in ckpt:
                module.load_state_dict(ckpt[key])
                loaded_any = True
                break

    # Some projects save nested model state dictionaries.
    if not loaded_any:
        for root_key in ("models", "model", "state_dict"):
            nested = ckpt.get(root_key)
            if not isinstance(nested, dict):
                continue

            for attr_name, key_candidates in candidates:
                module = getattr(trainer, attr_name, None)
                if module is None or not hasattr(module, "load_state_dict"):
                    continue

                for key in key_candidates:
                    if key in nested:
                        module.load_state_dict(nested[key])
                        loaded_any = True
                        break

    if not loaded_any:
        raise RuntimeError(
            "Could not infer the checkpoint format. "
            "Your CAMAPPOTrainer has no load_checkpoint() method and none of "
            "the common actor/critic keys were found. "
            "Please adapt load_checkpoint_into_trainer() to trainer.save_checkpoint()."
        )

    # Restore lambda when present.
    for key in ("lagrange_lambda", "lambda", "lambda_value"):
        if key in ckpt:
            value = float(ckpt[key])
            if hasattr(trainer, "lagrange_lambda"):
                obj = getattr(trainer, "lagrange_lambda")
                if torch.is_tensor(obj):
                    obj.data.fill_(value)
                else:
                    setattr(trainer, "lagrange_lambda", value)
            break

    return ckpt


# ---------------------------------------------------------------------
# One deterministic evaluation episode
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_one_seed(
    *,
    trainer: CAMAPPOTrainer,
    env,
    seed: int,
    episode_slots: int,
):
    obs, reset_info = env.reset(
        seed=seed,
        episode_length_slots=episode_slots,
    )

    B = int(env.B)
    U = int(env.U)

    current_es_action = None

    # Overall time series
    normalized_energy_hist = []
    power_w_hist = []
    service_deg_hist = []
    handover_rate_hist = []
    mlb_handover_rate_hist = []
    forced_es_handover_rate_hist = []
    outage_fraction_hist = []
    throughput_bps_hist = []
    demand_bps_hist = []
    demand_satisfaction_hist = []
    active_bs_count_hist = []

    # Full time-series history for later plotting.
    activation_hist = []
    bs_load_hist = []
    cio_hist = []
    delivered_rate_hist = []
    demand_rate_hist = []
    outage_mask_hist = []

    # Per-BS accumulators
    bs_on_sum = np.zeros(B, dtype=np.float64)
    bs_load_sum = np.zeros(B, dtype=np.float64)
    bs_load_count = np.zeros(B, dtype=np.int64)
    bs_cio_sum = np.zeros(B, dtype=np.float64)
    bs_cio_count = np.zeros(B, dtype=np.int64)
    bs_switches = np.zeros(B, dtype=np.int64)

    # Per-UE accumulators
    ue_delivered_sum = np.zeros(U, dtype=np.float64)
    ue_delivered_count = np.zeros(U, dtype=np.int64)

    ue_demand_sum = np.zeros(U, dtype=np.float64)
    ue_demand_count = np.zeros(U, dtype=np.int64)

    ue_outage_sum = np.zeros(U, dtype=np.float64)
    ue_outage_count = np.zeros(U, dtype=np.int64)

    ue_handover_count = np.zeros(U, dtype=np.int64)

    previous_activation = _vector(obs.get("activation"), B)
    previous_serving = None

    steps = 0

    for t in range(episode_slots):
        state = make_rl_state(obs, env)

        # --------------------------------------------------------------
        # ES: deterministic decision at epoch boundary
        # --------------------------------------------------------------
        if t % env.K == 0:
            eligible_mask = (
                np.asarray(obs["dwell_timer"]) >= (env.Tdwell - 1)
            )

            (
                current_es_action,
                _es_log_prob,
                _es_entropy,
                _es_value,
                _es_cost_value,
            ) = trainer.select_es_action(
                state,
                obs["activation"],
                eligible_mask,
                deterministic=True,
            )

        # --------------------------------------------------------------
        # MLB: deterministic decision every slot
        # --------------------------------------------------------------
        (
            _mlb_action_idx,
            mlb_cio_db,
            _mlb_log_prob,
            _mlb_entropy,
            _mlb_value,
            _cost_value,
        ) = trainer.select_mlb_action(
            state,
            deterministic=True,
        )

        next_obs, rewards, terminated, truncated, info = env.step(
            {
                "es": current_es_action if t % env.K == 0 else None,
                "mlb": mlb_cio_db,
            }
        )

        done = bool(terminated or truncated)
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}

        # ==============================================================
        # Overall metrics
        # ==============================================================
        normalized_energy = _find_value(
            metrics,
            info,
            keys=("normalized_energy", "energy_normalized"),
        )
        normalized_energy_hist.append(_scalar(normalized_energy))

        power_w = _find_value(
            metrics,
            info,
            keys=("energy_w", "power_w", "average_power_w", "total_power_w"),
        )
        power_w_hist.append(_scalar(power_w))

        service_deg = _find_value(
            metrics,
            info,
            keys=(
                "service_degradation",
                "service_degradation_ratio",
                "violation",
            ),
        )
        service_deg_hist.append(_scalar(service_deg))

        handover_rate = _find_value(
            metrics,
            info,
            keys=("handover_rate", "ho_rate"),
        )
        handover_rate_hist.append(_scalar(handover_rate))

        mlb_handover_rate = _find_value(
            metrics,
            info,
            keys=("mlb_handover_rate",),
        )
        mlb_handover_rate_hist.append(_scalar(mlb_handover_rate))

        forced_es_handover_rate = _find_value(
            metrics,
            info,
            keys=("forced_es_handover_rate",),
        )
        forced_es_handover_rate_hist.append(_scalar(forced_es_handover_rate))

        outage_fraction = _find_value(
            metrics,
            info,
            keys=("outage_fraction", "outage_rate"),
        )
        outage_fraction_hist.append(_scalar(outage_fraction))

        # ==============================================================
        # BS-level state
        # ==============================================================
        activation = _vector(
            _find_value(
                next_obs,
                info,
                keys=("activation", "bs_activation", "active_bs"),
            ),
            B,
        )

        if activation is None:
            activation = _vector(next_obs.get("activation"), B)

        if activation is not None:
            activation = activation.astype(np.float64)
            bs_on_sum += activation
            active_bs_count_hist.append(float(np.sum(activation)))
            activation_hist.append(activation.copy())

            if previous_activation is not None:
                bs_switches += (
                    activation.astype(np.int64)
                    != previous_activation.astype(np.int64)
                ).astype(np.int64)

            previous_activation = activation.copy()
        else:
            active_bs_count_hist.append(np.nan)
            activation_hist.append(np.full(B, np.nan, dtype=np.float64))

        bs_load = _vector(
            _find_value(
                next_obs,
                info,
                keys=("load", "bs_load", "cell_load"),
            ),
            B,
        )

        if bs_load is None:
            bs_load = _vector(next_obs.get("load"), B)

        if bs_load is not None:
            valid = np.isfinite(bs_load)
            bs_load_sum[valid] += bs_load[valid]
            bs_load_count[valid] += 1
            bs_load_hist.append(np.asarray(bs_load, dtype=np.float32).copy())
        else:
            bs_load_hist.append(np.full(B, np.nan, dtype=np.float32))

        cio_vec = _vector(mlb_cio_db, B)
        if cio_vec is None:
            cio_vec = _vector(
                _find_value(
                    next_obs,
                    info,
                    keys=("cio_db", "cio", "bs_cio_db"),
                ),
                B,
            )

        if cio_vec is not None:
            valid = np.isfinite(cio_vec)
            bs_cio_sum[valid] += cio_vec[valid]
            bs_cio_count[valid] += 1
            cio_hist.append(np.asarray(cio_vec, dtype=np.float32).copy())
        else:
            cio_hist.append(np.full(B, np.nan, dtype=np.float32))

        # ==============================================================
        # UE-level traffic
        # ==============================================================
        delivered = _vector(
            _find_value(
                info,
                keys=(
                    "delivered_rate_bps",
                    "ue_delivered_rate_bps",
                    "served_rate_bps",
                    "throughput_bps",
                ),
            ),
            U,
        )

        demand = _vector(
            _find_value(
                info,
                keys=(
                    "demand_bps",
                    "ue_demand_bps",
                    "traffic_demand_bps",
                ),
            ),
            U,
        )

        if delivered is not None:
            valid = np.isfinite(delivered)
            ue_delivered_sum[valid] += delivered[valid]
            ue_delivered_count[valid] += 1
            throughput_bps_hist.append(float(np.nansum(delivered)))
            delivered_rate_hist.append(np.asarray(delivered, dtype=np.float32).copy())
        else:
            throughput_bps_hist.append(np.nan)
            delivered_rate_hist.append(np.full(U, np.nan, dtype=np.float32))

        if demand is not None:
            valid = np.isfinite(demand)
            ue_demand_sum[valid] += demand[valid]
            ue_demand_count[valid] += 1
            demand_bps_hist.append(float(np.nansum(demand)))
            demand_rate_hist.append(np.asarray(demand, dtype=np.float32).copy())
        else:
            demand_bps_hist.append(np.nan)
            demand_rate_hist.append(np.full(U, np.nan, dtype=np.float32))

        if delivered is not None and demand is not None:
            ratio = np.divide(
                delivered,
                demand,
                out=np.ones_like(delivered, dtype=np.float64),
                where=demand > 0,
            )
            ratio = np.clip(ratio, 0.0, 1.0)
            demand_satisfaction_hist.append(float(np.nanmean(ratio)))
        else:
            demand_satisfaction_hist.append(np.nan)

        # ==============================================================
        # UE-level outage
        # ==============================================================
        outage_mask = _vector(
            _find_value(
                info,
                next_obs,
                keys=(
                    "outage_mask",
                    "ue_outage",
                    "outage",
                    "is_outage",
                ),
            ),
            U,
        )

        if outage_mask is not None:
            outage_mask = outage_mask.astype(np.float64)
            valid = np.isfinite(outage_mask)
            ue_outage_sum[valid] += outage_mask[valid]
            ue_outage_count[valid] += 1
            outage_mask_hist.append(outage_mask.astype(np.float32).copy())

            # Use UE mask to fill overall outage if scalar metric was absent.
            if np.isnan(outage_fraction_hist[-1]):
                outage_fraction_hist[-1] = float(np.nanmean(outage_mask))
        else:
            outage_mask_hist.append(np.full(U, np.nan, dtype=np.float32))

        # ==============================================================
        # UE-level handovers
        # ==============================================================
        serving = _vector(
            _find_value(
                info,
                next_obs,
                keys=(
                    "serving_cell",
                    "serving_bs",
                    "serving",
                    "serving_cell_idx",
                    "serving_bs_idx",
                ),
            ),
            U,
        )

        if serving is not None:
            serving = serving.astype(np.int64)

            if previous_serving is not None:
                # Ignore invalid/unserved cells if they are encoded as -1.
                valid_transition = (previous_serving >= 0) & (serving >= 0)
                ue_handover_count += (
                    valid_transition & (serving != previous_serving)
                ).astype(np.int64)

            previous_serving = serving.copy()

        obs = next_obs
        steps += 1

        if done:
            break

    if steps == 0:
        raise RuntimeError("Evaluation episode produced zero environment steps.")

    # -----------------------------------------------------------------
    # Episode-level calculations
    # -----------------------------------------------------------------
    mean_service_deg = _safe_nanmean(service_deg_hist)
    vmax = float(env.Vmax)

    constraint_violation = max(0.0, mean_service_deg - vmax)
    constraint_satisfied = float(mean_service_deg <= vmax)

    service_arr = np.asarray(service_deg_hist, dtype=np.float64)
    slot_violation_ratio = (
        float(np.nanmean(service_arr > vmax))
        if service_arr.size and not np.all(np.isnan(service_arr))
        else np.nan
    )

    total_delivered = float(np.nansum(ue_delivered_sum))
    total_demand = float(np.nansum(ue_demand_sum))

    if total_demand > 0:
        aggregate_demand_satisfaction = min(total_delivered / total_demand, 1.0)
    else:
        aggregate_demand_satisfaction = np.nan

    # Number of switching events reconstructed from activation history.
    switching_count = int(np.sum(bs_switches))

    # If environment exposes a trusted total, keep it for cross-checking.
    env_switching = _find_value(
        metrics if steps > 0 else {},
        info if steps > 0 else {},
        keys=("switching_count", "num_switches"),
    )
    env_switching_count = _scalar(env_switching)

    overall = {
        "seed": seed,
        "steps": steps,
        "normalized_energy": _safe_nanmean(normalized_energy_hist),
        "average_power_w": _safe_nanmean(power_w_hist),
        "service_degradation": mean_service_deg,
        "constraint_violation": constraint_violation,
        "constraint_satisfied": constraint_satisfied,
        "slot_violation_ratio": slot_violation_ratio,
        "outage_fraction": _safe_nanmean(outage_fraction_hist),
        "total_throughput_mbps": (
            _safe_nanmean(throughput_bps_hist) / 1e6
        ),
        "demand_satisfaction_ratio": (
            aggregate_demand_satisfaction
            if np.isfinite(aggregate_demand_satisfaction)
            else _safe_nanmean(demand_satisfaction_hist)
        ),
        "handover_rate": _safe_nanmean(handover_rate_hist),
        "mlb_handover_rate": _safe_nanmean(mlb_handover_rate_hist),
        "forced_es_handover_rate": _safe_nanmean(forced_es_handover_rate_hist),
        "switching_count": switching_count,
        "env_switching_count": env_switching_count,
        "average_active_bs_count": _safe_nanmean(active_bs_count_hist),
    }

    # -----------------------------------------------------------------
    # Per-BS dataframe
    # -----------------------------------------------------------------
    per_bs_rows = []

    for b in range(B):
        per_bs_rows.append(
            {
                "seed": seed,
                "bs": b,
                "on_ratio": bs_on_sum[b] / steps,
                "average_load": (
                    bs_load_sum[b] / bs_load_count[b]
                    if bs_load_count[b] > 0
                    else np.nan
                ),
                "average_cio_db": (
                    bs_cio_sum[b] / bs_cio_count[b]
                    if bs_cio_count[b] > 0
                    else np.nan
                ),
                "number_of_switches": int(bs_switches[b]),
            }
        )

    # -----------------------------------------------------------------
    # Per-UE dataframe
    # -----------------------------------------------------------------
    per_ue_rows = []

    for u in range(U):
        avg_delivered = (
            ue_delivered_sum[u] / ue_delivered_count[u]
            if ue_delivered_count[u] > 0
            else np.nan
        )

        avg_demand = (
            ue_demand_sum[u] / ue_demand_count[u]
            if ue_demand_count[u] > 0
            else np.nan
        )

        if np.isfinite(avg_delivered) and np.isfinite(avg_demand) and avg_demand > 0:
            ue_satisfaction = min(avg_delivered / avg_demand, 1.0)
        else:
            ue_satisfaction = np.nan

        ue_outage_ratio = (
            ue_outage_sum[u] / ue_outage_count[u]
            if ue_outage_count[u] > 0
            else np.nan
        )

        per_ue_rows.append(
            {
                "seed": seed,
                "ue": u,
                "average_delivered_rate_mbps": (
                    avg_delivered / 1e6 if np.isfinite(avg_delivered) else np.nan
                ),
                "average_demand_mbps": (
                    avg_demand / 1e6 if np.isfinite(avg_demand) else np.nan
                ),
                "demand_satisfaction_ratio": ue_satisfaction,
                "outage_ratio": ue_outage_ratio,
                "handover_count": int(ue_handover_count[u]),
            }
        )

    history = {
        "normalized_energy": np.asarray(normalized_energy_hist, dtype=np.float32),
        "power_w": np.asarray(power_w_hist, dtype=np.float32),
        "service_degradation": np.asarray(service_deg_hist, dtype=np.float32),
        "handover_rate": np.asarray(handover_rate_hist, dtype=np.float32),
        "mlb_handover_rate": np.asarray(mlb_handover_rate_hist, dtype=np.float32),
        "forced_es_handover_rate": np.asarray(forced_es_handover_rate_hist, dtype=np.float32),
        "outage_fraction": np.asarray(outage_fraction_hist, dtype=np.float32),
        "throughput_bps": np.asarray(throughput_bps_hist, dtype=np.float32),
        "demand_bps": np.asarray(demand_bps_hist, dtype=np.float32),
        "demand_satisfaction": np.asarray(demand_satisfaction_hist, dtype=np.float32),
        "active_bs_count": np.asarray(active_bs_count_hist, dtype=np.float32),
        "activation": np.asarray(activation_hist, dtype=np.float32),
        "bs_load": np.asarray(bs_load_hist, dtype=np.float32),
        "cio_db": np.asarray(cio_hist, dtype=np.float32),
        "delivered_rate_bps": np.asarray(delivered_rate_hist, dtype=np.float32),
        "demand_rate_bps": np.asarray(demand_rate_hist, dtype=np.float32),
        "outage_mask": np.asarray(outage_mask_hist, dtype=np.float32),
        "seed": np.asarray([seed], dtype=np.int64),
        "steps": np.asarray([steps], dtype=np.int64),
        "K": np.asarray([env.K], dtype=np.int64),
        "Vmax": np.asarray([env.Vmax], dtype=np.float32),
        "B": np.asarray([env.B], dtype=np.int64),
        "U": np.asarray([env.U], dtype=np.int64),
    }

    return overall, per_bs_rows, per_ue_rows, history


# ---------------------------------------------------------------------
# Aggregate across evaluation seeds
# ---------------------------------------------------------------------
def make_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "normalized_energy",
        "average_power_w",
        "service_degradation",
        "constraint_violation",
        "constraint_satisfied",
        "slot_violation_ratio",
        "outage_fraction",
        "total_throughput_mbps",
        "demand_satisfaction_ratio",
        "handover_rate",
        "switching_count",
        "average_active_bs_count",
    ]

    rows = []

    for metric in metrics:
        values = pd.to_numeric(per_seed_df[metric], errors="coerce")

        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=0),
                "min": values.min(),
                "max": values.max(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------
def evaluate(
    *,
    checkpoint_path: str | Path,
    eval_seeds: list[int],
    config_path: str | Path | None = None,
    episode_slots: int | None = None,
    output_dir: str | Path = "results/evaluation",
) -> None:
    root = Path(__file__).resolve().parents[1]

    if config_path is None:
        config_path = root / "env" / "config.yaml"

    env = load_env(config_path)
    cfg = env.cfg
    train_cfg = cfg["accord_training"]

    if episode_slots is None:
        def find_config_value(node, target_key):
            if isinstance(node, dict):
                if target_key in node:
                    return node[target_key]
                for value in node.values():
                    found = find_config_value(value, target_key)
                    if found is not None:
                        return found
            elif isinstance(node, (list, tuple)):
                for value in node:
                    found = find_config_value(value, target_key)
                    if found is not None:
                        return found
            return None

        diurnal_cycle_slots = find_config_value(cfg, "diurnal_cycle_slots")

        if diurnal_cycle_slots is None:
            raise KeyError(
                "`diurnal_cycle_slots` was not found in config.yaml. "
                "Evaluation episode length is intended to cover one full "
                "diurnal traffic cycle."
            )

        episode_slots = int(diurnal_cycle_slots)

    if episode_slots % env.K != 0:
        warnings.warn(
            f"episode_slots={episode_slots} is not divisible by K={env.K}. "
            "The last ES epoch may be incomplete."
        )

    set_seed(0)

    trainer = CAMAPPOTrainer(
        state_dim=env.state_dim,
        num_cells=env.B,
        cio_values=env.cio_values,
        config=cfg,
        epoch_slots=env.K,
        vmax=env.Vmax,
    )

    load_checkpoint_into_trainer(
        trainer=trainer,
        checkpoint_path=checkpoint_path,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {trainer.device}")
    print(
        f"State dim: {env.state_dim}, B={env.B}, U={env.U}, "
        f"K={env.K}, Tdwell={env.Tdwell}, Vmax={env.Vmax}"
    )
    print(f"Evaluation slots: {episode_slots}")
    print(f"Evaluation seeds: {eval_seeds}")
    print()

    overall_rows = []
    per_bs_rows = []
    per_ue_rows = []

    for i, eval_seed in enumerate(eval_seeds, start=1):
        overall, bs_rows, ue_rows, history = evaluate_one_seed(
            trainer=trainer,
            env=env,
            seed=eval_seed,
            episode_slots=episode_slots,
        )

        overall_rows.append(overall)
        per_bs_rows.extend(bs_rows)
        per_ue_rows.extend(ue_rows)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        history_path = output_dir_path / f"accord_history_seed{eval_seed}.npz"
        np.savez_compressed(history_path, **history)

        print(
            f"[Eval {i:02d}/{len(eval_seeds):02d} | seed={eval_seed}] "
            f"E={overall['normalized_energy']:.4f} | "
            f"P={overall['average_power_w']:.2f} W | "
            f"V={overall['service_degradation']:.4f} | "
            f"CV={overall['constraint_violation']:.4f} | "
            f"SAT={int(overall['constraint_satisfied'])} | "
            f"Out={overall['outage_fraction']:.4f} | "
            f"Thr={overall['total_throughput_mbps']:.2f} Mbps | "
            f"DSR={overall['demand_satisfaction_ratio']:.4f} | "
            f"HO={overall['handover_rate']:.4f} | "
            f"SW={overall['switching_count']} | "
            f"ActiveBS={overall['average_active_bs_count']:.2f}"
        )

    per_seed_df = pd.DataFrame(overall_rows)
    per_bs_df = pd.DataFrame(per_bs_rows)
    per_ue_df = pd.DataFrame(per_ue_rows)
    summary_df = make_summary(per_seed_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_path = output_dir / "accord_per_seed.csv"
    per_bs_path = output_dir / "accord_per_bs.csv"
    per_ue_path = output_dir / "accord_per_ue.csv"
    summary_path = output_dir / "accord_summary.csv"
    metadata_path = output_dir / "accord_evaluation_metadata.json"

    per_seed_df.to_csv(per_seed_path, index=False)
    per_bs_df.to_csv(per_bs_path, index=False)
    per_ue_df.to_csv(per_ue_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "evaluation_slots": int(episode_slots),
        "evaluation_seeds": [int(s) for s in eval_seeds],
        "B": int(env.B),
        "U": int(env.U),
        "K": int(env.K),
        "Tdwell": int(env.Tdwell),
        "Vmax": float(env.Vmax),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("Evaluation summary")
    print("=" * 72)

    display_df = summary_df.copy()
    for column in ("mean", "std", "min", "max"):
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.6f}" if pd.notna(x) else "nan"
        )

    print(display_df.to_string(index=False))
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_seed_path}")
    print(f"Saved: {per_bs_path}")
    print(f"Saved: {per_ue_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained ACCORD checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to env/config.yaml. Default: repository env/config.yaml",
    )
    parser.add_argument(
        "--eval_seeds",
        type=int,
        nargs="+",
        default=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        help="Unseen environment seeds used for evaluation.",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=None,
        help=(
            "Evaluation episode length override. Default: "
            "config.yaml의 diurnal_cycle_slots 값."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        eval_seeds=args.eval_seeds,
        config_path=args.config,
        episode_slots=args.slots,
        output_dir=args.output_dir,
    )
