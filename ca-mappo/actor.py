"""
    Actor networks for ACCORD.

    ES actor:
        pi_E(s | o_E) = prod_b Bernoulli(s_b; p_b)
        - one Bernoulli logit per BS
        - ineligible BSs (dwell constraint) keep their current state
        - ineligible BSs contribute zero log-probability and zero entropy

    MLB actor:
        pi_M(theta | o_M) = prod_b Categorical(theta_b; q_b)
        - one |Theta|-way categorical distribution per BS

    The paper specifies MLPs and hidden size 128, but does not fix the number of hidden layers or activation. 
    We use a configurable two-layer Tanh MLP by default, a standard PPO choice.
"""

import __future__ as annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

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


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Iterable[int] = (128, 128),
    activation: str = "tanh",
) -> nn.Sequential:
    """Create an MLP used by an ACCORD actor."""
    act_cls = _activation(activation)
    layers: list[nn.Module] = []
    prev = int(input_dim)

    for hidden in hidden_sizes:
        hidden = int(hidden)
        layers += [nn.Linear(prev, hidden), act_cls()]
        prev = hidden

    layers.append(nn.Linear(prev, int(output_dim)))
    net = nn.Sequential(*layers)

    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.constant_(module.bias, 0.0)

    # Small final layer => near-uniform initial policy.
    final = net[-1]
    assert isinstance(final, nn.Linear)
    nn.init.orthogonal_(final.weight, gain=0.01)
    nn.init.constant_(final.bias, 0.0)
    return net

def _ensure_batch(x: Tensor) -> Tuple[Tensor, bool]:
    """Convert [D] -> [1,D]. Return tensor and whether it was unbatched."""
    if x.dim() == 1:
        return x.unsqueeze(0), True
    if x.dim() != 2:
        raise ValueError(f"Expected [D] or [N,D], got shape {tuple(x.shape)}")
    return x, False


class ESActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_cells: int,
        hidden_sizes: Iterable[int] = (128, 128),
        activation: str = "tanh",
        ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_cells = int(num_cells)
        self.net = build_mlp(obs_dim, num_cells, hidden_sizes, activation)

    def forward(self, obs: Tensor) -> Tensor:
        """Return Bernoulli logits, shape [N,B] (or [B] for unbatched input)."""
        obs_b, was_unbatched = _ensure_batch(obs)
        logits = self.net(obs_b)
        return logits.squeeze(0) if was_unbatched else logits

    @torch.no_grad()
    def act(
        self,
        obs: Tensor, 
        current_state: Tensor,
        eligible_mask: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        obs_b, was_unbatched = _ensure_batch(obs)
        current_b, _ = _ensure_batch(current_state.float())
        mask_b, _ = _ensure_batch(eligible_mask.bool())

        if current_b.shape != (obs_b.shape[0], self.num_cells):
            raise ValueError(
                f"current_state must be [N,{self.num_cells}], got {tuple(current_b.shape)}"
            )
        if mask_b.shape != (obs_b.shape[0], self.num_cells):
            raise ValueError(
                f"eligible_mask must be [N,{self.num_cells}], got {tuple(mask_b.shape)}"
            )

        logits = self.net(obs_b)
        dist = Bernoulli(logits=logits)

        if deterministic:
            proposed = (logits >= 0.0).float()
        else:
            proposed = dist.sample()

        action = torch.where(mask_b, proposed, current_b)
        per_cell_logp = dist.log_prob(action)
        per_cell_entropy = dist.entropy()
        mask_f = mask_b.float()
        log_prob = (per_cell_logp * mask_f).sum(dim=-1)
        entropy = (per_cell_entropy * mask_f).sum(dim=-1)

        action = action.long()
        if was_unbatched:
            return action.squeeze(0), log_prob.squeeze(0), entropy.squeeze(0)
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: Tensor,
        actions: Tensor,
        eligible_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        obs_b, was_unbatched = _ensure_batch(obs)
        actions_b, _ = _ensure_batch(actions.float())
        mask_b, _ = _ensure_batch(eligible_mask.bool())

        logits = self.net(obs_b)
        dist = Bernoulli(logits=logits)
        mask_f = mask_b.float()

        log_prob = (dist.log_prob(actions_b) * mask_f).sum(dim=-1)
        entropy = (dist.entropy() * mask_f).sum(dim=-1)

        if was_unbatched:
            return log_prob.squeeze(0), entropy.squeeze(0)
        return log_prob, entropy

class MLBActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_cells: int,
        cio_values: Iterable[float],
        hidden_sizes: Iterable[int] = (128, 128),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_cells = int(num_cells)
        cio_tensor = torch.as_tensor(list(cio_values), dtype=torch.float32)
        if cio_tensor.ndim != 1 or cio_tensor.numel() < 2:
            raise ValueError(f"cio_values must be 1D, got shape {tuple(cio_tensor.shape)}")

        self.register_buffer("cio_values", cio_tensor)
        self.num_cio = int(cio_tensor.numel())
        
        self.net = build_mlp(self.obs_dim, self.num_cells * self.num_cio, hidden_sizes, activation)

    def forward(self, obs: Tensor) -> Tensor:
        """Return per-BS logits, shape [N,B,A] (or [B,A] for unbatched input)."""
        obs_b, was_unbatched = _ensure_batch(obs)
        logits = self.net(obs_b).view(-1, self.num_cells, self.num_actions)
        return logits.squeeze(0) if was_unbatched else logits

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        obs_b, was_unbatched = _ensure_batch(obs)
        logits = self.net(obs_b).view(-1, self.num_cells, self.num_cio)
        dist = Categorical(logits=logits)

        if deterministic:
            action_idx = logits.argmax(dim=-1)
        else:
            action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        cio_db = self.cio_values[action_idx]

        if was_unbatched:
            return (
                action_idx.squeeze(0), 
                cio_db.squeeze(0), 
                log_prob.squeeze(0), 
                entropy.squeeze(0)
            )
        return action_idx, cio_db, log_prob, entropy

    def evaluate_actions(
        self,
        obs: Tensor,
        action_idx: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Recompute PPO log-probability and entropy for stored MLB actions."""
        obs_b, was_unbatched = _ensure_batch(obs)
        action_b, _ = _ensure_batch(action_idx.long())
        
        logits = self.net(obs_b).view(-1, self.num_cells, self.num_cio)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action_b).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        if was_unbatched:
            return log_prob.squeeze(0), entropy.squeeze(0)
        return log_prob, entropy