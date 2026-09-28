"""Utilities for ACCORD training."""
from __future__ import annotations

import random
from typing import Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_normalized_observation(
    obs: Mapping[str, np.ndarray],
    *,
    cio_abs_max: float,
    dwell_time: int,
    nominal_demand_bps: float,
    num_ues: int,
) -> np.ndarray:
    """Convert the raw environment state into a stable NN input.
        load = rho / (1 + rho)
        cio = cio / max(abs(cio_min), abs(cio_max))
        demand = demand / (nominal_demand * num_ues)
        dwell = dwell / T_dwell
    """
    load = np.asarray(obs["load"], dtype=np.float32)
    activation = np.asarray(obs["activation"], dtype=np.float32)
    cio = np.asarray(obs["cio_db"], dtype=np.float32)
    demand_hist = np.asarray(obs["demand_history_bps"], dtype=np.float32)
    dwell = np.asarray(obs["dwell_timer"], dtype=np.float32)

    load_n = np.maximum(load, 0.0) / (1.0 + np.maximum(load, 0.0))

    cio_scale = max(float(cio_abs_max), 1.0)
    cio_n = np.clip(cio / cio_scale, -1.0, 1.0)

    demand_scale = max(float(nominal_demand_bps) * int(num_ues), 1.0)
    demand_n = np.clip(demand_hist / demand_scale, 0.0, 1.0)

    dwell_scale = max(float(dwell_time), 1.0)
    dwell_n = np.clip(dwell / dwell_scale, 0.0, 1.0)

    return np.concatenate(
        [
            load_n.ravel(),
            activation.ravel(),
            cio_n.ravel(),
            demand_n.ravel(),
            dwell_n.ravel(),
        ]
    ).astype(np.float32)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation for one trajectory."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values and dones must have identical shapes")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            value_next = float(next_value)
        else:
            value_next = float(values[t + 1])

        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * value_next * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size <= 1:
        return np.zeros_like(x)
    return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)


def epoch_average(slot_values: np.ndarray, epoch_slots: int) -> np.ndarray:
    slot_values = np.asarray(slot_values, dtype=np.float32)
    if len(slot_values) % int(epoch_slots) != 0:
        raise ValueError(
            f"slot_values length {len(slot_values)} must be divisible by K={epoch_slots}"
        )
    return slot_values.reshape(-1, int(epoch_slots)).mean(axis=1).astype(np.float32)
