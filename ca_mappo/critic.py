"""Centralized critics for ACCORD.

ACCORD trains three scalar critics on the global state x_t:
    1) ES value critic   V^E_psiE
    2) MLB value critic  V^M_psiM
    3) Cost critic       V^C_psiC

The three critics intentionally do NOT share their regression target.
The cost critic estimates discounted service degradation V_t and is used only in the ES constrained update.
"""

from __future__ import annotations

from typing import Iterable
import torch
import torch.nn as nn

Tensor = torch.Tensor

def _activation(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported activation: {name}")

def build_value_mlp(
    state_dim: int,
    hidden_sizes: Iterable[int] = (128, 128),
    activation: str = "tanh",
) -> nn.Sequential:
    act_cls = _activation(activation)
    layers: list[nn.Module] = []
    prev = int(state_dim)

    for hidden in hidden_sizes:
        hidden = int(hidden)
        layers += [nn.Linear(prev, hidden), act_cls()]
        prev = hidden

    layers.append(nn.Linear(prev, 1))
    net = nn.Sequential(*layers)

    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.constant_(module.bias, 0.0)

    final = net[-1]
    assert isinstance(final, nn.Linear)
    nn.init.orthogonal_(final.weight, gain=1.0)
    nn.init.constant_(final.bias, 0.0)
    return net

class ValueCritic(nn.Module):
    """Scalar centralized value function V(x)."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.net = build_value_mlp(
            state_dim=self.state_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, state: Tensor) -> Tensor:
        """Return scalar value(s): [D] -> [], [N,D] -> [N]."""
        if state.dim() == 1:
            return self.net(state.unsqueeze(0)).squeeze(0).squeeze(-1)
        if state.dim() != 2:
            raise ValueError(f"Expected [D] or [N,D], got shape {tuple(state.shape)}")
        return self.net(state).squeeze(-1)


class ESValueCritic(ValueCritic):
    """V^E(x_{kK}): predicts epoch-scale ES reward return."""


class MLBValueCritic(ValueCritic):
    """V^M(x_t): predicts slot-scale MLB reward return."""


class CostCritic(ValueCritic):
    """V^C(x_t): predicts slot-scale service-degradation cost return."""
