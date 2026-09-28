"""Rollout buffers for ACCORD.

Two buffers are used because the agents act on different time scales:
- MLB: one transition per environment slot.
- ES: one transition per ES epoch (K slots).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch

Tensor = torch.Tensor


@dataclass
class MLBRolloutBuffer:
    '''
        reward = -V - wh*H
        cost = V
        value = V_m(x_t)
        cost_value = V_c(x_t)
    '''
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    cost_values: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)

    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None
    cost_advantages: np.ndarray | None = None
    cost_returns: np.ndarray | None = None

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        old_log_prob: float,
        reward: float,
        cost: float,
        value: float,
        cost_value: float,
        done: bool,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self.old_log_probs.append(float(old_log_prob))
        self.rewards.append(float(reward))
        self.costs.append(float(cost))
        self.values.append(float(value))
        self.cost_values.append(float(cost_value))
        self.dones.append(float(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def set_training_targets(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        cost_advantages: np.ndarray,
        cost_returns: np.ndarray,
    ) -> None:
        n = len(self)
        for name, arr in {
            "advantages": advantages,
            "returns": returns,
            "cost_advantages": cost_advantages,
            "cost_returns": cost_returns,
        }.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
            setattr(self, name, arr)

    def minibatches(
        self,
        minibatch_size: int,
        device: torch.device,
        shuffle: bool = True,
    ) -> Iterator[dict[str, Tensor]]:
        if any(x is None for x in (self.advantages, self.returns, self.cost_advantages, self.cost_returns)):
            raise RuntimeError("Call set_training_targets() before minibatches().")

        n = len(self)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int64)
        old_log_probs = np.asarray(self.old_log_probs, dtype=np.float32)

        for start in range(0, n, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield {
                "states": torch.as_tensor(states[idx], device=device),
                "actions": torch.as_tensor(actions[idx], device=device),
                "old_log_probs": torch.as_tensor(old_log_probs[idx], device=device),
                "advantages": torch.as_tensor(self.advantages[idx], device=device),
                "returns": torch.as_tensor(self.returns[idx], device=device),
                "cost_advantages": torch.as_tensor(self.cost_advantages[idx], device=device),
                "cost_returns": torch.as_tensor(self.cost_returns[idx], device=device),
            }


@dataclass
class ESRolloutBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    eligible_masks: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)

    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None
    penalized_advantages: np.ndarray | None = None

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        old_log_prob: float,
        eligible_mask: np.ndarray,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self.old_log_probs.append(float(old_log_prob))
        self.eligible_masks.append(np.asarray(eligible_mask, dtype=bool).copy())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def set_training_targets(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        penalized_advantages: np.ndarray,
    ) -> None:
        n = len(self)
        for name, arr in {
            "advantages": advantages,
            "returns": returns,
            "penalized_advantages": penalized_advantages,
        }.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
            setattr(self, name, arr)

    def minibatches(
        self,
        minibatch_size: int,
        device: torch.device,
        shuffle: bool = True,
    ) -> Iterator[dict[str, Tensor]]:
        if any(x is None for x in (self.advantages, self.returns, self.penalized_advantages)):
            raise RuntimeError("Call set_training_targets() before minibatches().")

        n = len(self)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int64)
        old_log_probs = np.asarray(self.old_log_probs, dtype=np.float32)
        eligible_masks = np.asarray(self.eligible_masks, dtype=bool)

        for start in range(0, n, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield {
                "states": torch.as_tensor(states[idx], device=device),
                "actions": torch.as_tensor(actions[idx], device=device),
                "old_log_probs": torch.as_tensor(old_log_probs[idx], device=device),
                "eligible_masks": torch.as_tensor(eligible_masks[idx], device=device),
                "advantages": torch.as_tensor(self.advantages[idx], device=device),
                "returns": torch.as_tensor(self.returns[idx], device=device),
                "penalized_advantages": torch.as_tensor(self.penalized_advantages[idx], device=device),
            }
