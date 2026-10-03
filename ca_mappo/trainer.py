"""CA-MAPPO / ACCORD trainer.

The key ACCORD features implemented here are:
1. Separate ES, MLB and cost critics.
2. Slot-scale MLB/cost GAE and epoch-scale ES GAE.
3. A dual variable applied only to the ES policy advantage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .actor import ESActor, MLBActor
from .buffer import ESRolloutBuffer, MLBRolloutBuffer
from .critic import CostCritic, ESValueCritic, MLBValueCritic
from .utils import compute_gae, epoch_average, normalize


class CAMAPPOTrainer:
    def __init__(
        self,
        *,
        state_dim: int,
        num_cells: int,
        cio_values: np.ndarray,
        config: dict[str, Any],
        epoch_slots: int,
        vmax: float,
        device: str | torch.device | None = None,
    ) -> None:
        train_cfg = config["accord_training"]

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        hidden = int(train_cfg.get("hidden_size", 128))
        hidden_sizes = (hidden, hidden)

        self.es_actor = ESActor(state_dim, num_cells, hidden_sizes).to(self.device)
        self.mlb_actor = MLBActor(state_dim, num_cells, cio_values, hidden_sizes).to(self.device)
        self.es_critic = ESValueCritic(state_dim, hidden_sizes).to(self.device)
        self.mlb_critic = MLBValueCritic(state_dim, hidden_sizes).to(self.device)
        self.cost_critic = CostCritic(state_dim, hidden_sizes).to(self.device)

        actor_lr = float(train_cfg["actor_learning_rate"])
        critic_lr = float(train_cfg["critic_learning_rate"])

        self.es_actor_opt = torch.optim.Adam(self.es_actor.parameters(), lr=actor_lr)
        self.mlb_actor_opt = torch.optim.Adam(self.mlb_actor.parameters(), lr=actor_lr)
        self.es_critic_opt = torch.optim.Adam(self.es_critic.parameters(), lr=critic_lr)
        self.mlb_critic_opt = torch.optim.Adam(self.mlb_critic.parameters(), lr=critic_lr)
        self.cost_critic_opt = torch.optim.Adam(self.cost_critic.parameters(), lr=critic_lr)

        self.gamma = float(train_cfg["gamma"])
        self.gamma_es = float(train_cfg["gamma_es"])
        self.gae_lambda = float(train_cfg["gae_lambda"])
        self.clip_epsilon = float(train_cfg["clip_epsilon"])
        self.entropy_beta = float(train_cfg["entropy_beta"])
        self.update_iterations = int(train_cfg["update_iterations"])
        self.minibatch_mlb = int(train_cfg["minibatch_mlb"])
        self.minibatch_es = int(train_cfg["minibatch_es"])
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
        self.value_loss_coef = float(train_cfg.get("value_loss_coef", 1.0))

        self.K = int(epoch_slots)
        self.vmax = float(vmax)
        self.dual_step_size = float(train_cfg["dual_step_size"])
        self.smoothing_kappa = float(train_cfg["smoothing_kappa"])
        self.lagrange_lambda = 0.0
        self.cost_ema = 0.0

    @torch.no_grad()
    def select_es_action(
        self,
        state: np.ndarray,
        current_activation: np.ndarray,
        eligible_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        current_t = torch.as_tensor(current_activation, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(eligible_mask, dtype=torch.bool, device=self.device)

        action, log_prob, entropy = self.es_actor.act(
            state_t, current_t, mask_t, deterministic=deterministic
        )
        value = self.es_critic(state_t)
        return (
            action.cpu().numpy(),
            float(log_prob.item()),
            float(entropy.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def select_mlb_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_idx, cio_db, log_prob, entropy = self.mlb_actor.act(
            state_t, deterministic=deterministic
        )
        value = self.mlb_critic(state_t)
        cost_value = self.cost_critic(state_t)
        return (
            action_idx.cpu().numpy(),
            cio_db.cpu().numpy(),
            float(log_prob.item()),
            float(entropy.item()),
            float(value.item()),
            float(cost_value.item()),
        )

    def _prepare_targets(
        self,
        mlb_buffer: MLBRolloutBuffer,
        es_buffer: ESRolloutBuffer,
        *,
        next_mlb_value: float,
        next_cost_value: float,
        next_es_value: float,
    ) -> dict[str, float]:
        mlb_adv, mlb_returns = compute_gae(
            np.asarray(mlb_buffer.rewards),
            np.asarray(mlb_buffer.values),
            np.asarray(mlb_buffer.dones),
            next_mlb_value,
            self.gamma,
            self.gae_lambda,
        )
        cost_adv, cost_returns = compute_gae(
            np.asarray(mlb_buffer.costs),
            np.asarray(mlb_buffer.cost_values),
            np.asarray(mlb_buffer.dones),
            next_cost_value,
            self.gamma,
            self.gae_lambda,
        )
        es_adv, es_returns = compute_gae(
            np.asarray(es_buffer.rewards),
            np.asarray(es_buffer.values),
            np.asarray(es_buffer.dones),
            next_es_value,
            self.gamma_es,
            self.gae_lambda,
        )

        if len(cost_adv) != len(es_buffer) * self.K:
            raise ValueError(
                "For the current single-environment implementation, rollout length "
                "must contain complete K-slot ES epochs."
            )

        cost_adv_epoch = epoch_average(cost_adv, self.K)

        # Normalize advantages
        mlb_adv_n = normalize(mlb_adv)
        es_adv_n = normalize(es_adv)
        cost_adv_epoch_n = normalize(cost_adv_epoch)

        # Normalize return targets
        mlb_returns_n = normalize(mlb_returns)
        es_returns_n = normalize(es_returns)
        cost_returns_n = normalize(cost_returns)

        penalized_es_adv = (
            es_adv_n - self.lagrange_lambda * cost_adv_epoch_n
        ) / (1.0 + self.lagrange_lambda)

        lambda_cost_adv = self.lagrange_lambda * cost_adv_epoch_n
        constraint_dominance_frac = float(np.mean(np.abs(lambda_cost_adv) > np.abs(es_adv_n)))
        sign_flip_frac = float(np.mean(np.sign(lambda_cost_adv) != np.sign(es_adv_n)))


        mlb_buffer.set_training_targets(
            advantages=mlb_adv_n,
            returns=mlb_returns_n,
            cost_advantages=cost_adv,
            cost_returns=cost_returns_n,
        )
        es_buffer.set_training_targets(
            advantages=es_adv_n,
            returns=es_returns_n,
            penalized_advantages=penalized_es_adv,
        )

        return {
            "mean_mlb_adv": float(np.mean(mlb_adv)),
            "mean_es_adv": float(np.mean(es_adv)),
            "mean_cost_adv": float(np.mean(cost_adv)),
            
            "es_adv_norm_abs_mean": float(np.mean(np.abs(es_adv_n))),
            "es_adv_norm_min": float(np.min(es_adv_n)),
            "es_adv_norm_max": float(np.max(es_adv_n)),
            "cost_adv_epoch_norm_abs_mean": float(np.mean(np.abs(cost_adv_epoch_n))),
            "cost_adv_epoch_norm_min": float(np.min(cost_adv_epoch_n)),
            "cost_adv_epoch_norm_max": float(np.max(cost_adv_epoch_n)),
            "lambda_cost_adv_abs_mean": float(np.mean(np.abs(lambda_cost_adv))),
            "lambda_cost_adv_min": float(np.min(lambda_cost_adv)),
            "lambda_cost_adv_max": float(np.max(lambda_cost_adv)),
            "penalized_es_adv_abs_mean": float(np.mean(np.abs(penalized_es_adv))),
            "penalized_es_adv_min": float(np.min(penalized_es_adv)),
            "penalized_es_adv_max": float(np.max(penalized_es_adv)),
            "constraint_dominance_frac": constraint_dominance_frac,
            "es_adv_sign_flip_frac": sign_flip_frac,
        }

    def update_dual(self, costs: list[float]) -> tuple[float, float]:
        mean_cost = float(np.mean(costs)) if costs else 0.0
        self.cost_ema = (
            (1.0 - self.smoothing_kappa) * self.cost_ema
            + self.smoothing_kappa * mean_cost
        )
        self.lagrange_lambda = max(
            0.0,
            self.lagrange_lambda
            + self.dual_step_size * (self.cost_ema - self.vmax),
        )
        return mean_cost, self.lagrange_lambda

    def _update_mlb(self, buffer: MLBRolloutBuffer) -> dict[str, float]:
        actor_losses, critic_losses, cost_losses, entropies = [], [], [], []

        for _ in range(self.update_iterations):
            for batch in buffer.minibatches(self.minibatch_mlb, self.device):
                states = batch["states"]

                # MLB actor
                new_logp, entropy = self.mlb_actor.evaluate_actions(states, batch["actions"])
                ratio = torch.exp(new_logp - batch["old_log_probs"])
                adv = batch["advantages"]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * adv
                actor_loss = -torch.min(unclipped, clipped).mean() - self.entropy_beta * entropy.mean()

                self.mlb_actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mlb_actor.parameters(), self.max_grad_norm)
                self.mlb_actor_opt.step()

                # MLB reward critic
                value_pred = self.mlb_critic(states)
                critic_loss = F.mse_loss(value_pred, batch["returns"])
                self.mlb_critic_opt.zero_grad(set_to_none=True)
                (self.value_loss_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.mlb_critic.parameters(), self.max_grad_norm)
                self.mlb_critic_opt.step()

                # Service-degradation cost critic
                cost_pred = self.cost_critic(states)
                cost_loss = F.mse_loss(cost_pred, batch["cost_returns"])
                self.cost_critic_opt.zero_grad(set_to_none=True)
                (self.value_loss_coef * cost_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.cost_critic_opt.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                cost_losses.append(float(cost_loss.item()))
                entropies.append(float(entropy.mean().item()))

        return {
            "mlb_actor_loss": float(np.mean(actor_losses)),
            "mlb_critic_loss": float(np.mean(critic_losses)),
            "cost_critic_loss": float(np.mean(cost_losses)),
            "mlb_entropy": float(np.mean(entropies)),
        }

    def _update_es(self, buffer: ESRolloutBuffer) -> dict[str, float]:
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.update_iterations):
            for batch in buffer.minibatches(self.minibatch_es, self.device):
                states = batch["states"]

                new_logp, entropy = self.es_actor.evaluate_actions(
                    states,
                    batch["actions"],
                    batch["eligible_masks"],
                )
                ratio = torch.exp(new_logp - batch["old_log_probs"])
                adv = batch["penalized_advantages"]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * adv
                actor_loss = -torch.min(unclipped, clipped).mean() - self.entropy_beta * entropy.mean()

                self.es_actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.es_actor.parameters(), self.max_grad_norm)
                self.es_actor_opt.step()

                value_pred = self.es_critic(states)
                critic_loss = F.mse_loss(value_pred, batch["returns"])
                self.es_critic_opt.zero_grad(set_to_none=True)
                (self.value_loss_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.es_critic.parameters(), self.max_grad_norm)
                self.es_critic_opt.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.mean().item()))

        return {
            "es_actor_loss": float(np.mean(actor_losses)),
            "es_critic_loss": float(np.mean(critic_losses)),
            "es_entropy": float(np.mean(entropies)),
        }

    def update(
        self,
        mlb_buffer: MLBRolloutBuffer,
        es_buffer: ESRolloutBuffer,
        *,
        next_mlb_value: float = 0.0,
        next_cost_value: float = 0.0,
        next_es_value: float = 0.0,
    ) -> dict[str, float]:
        if len(mlb_buffer) == 0 or len(es_buffer) == 0:
            raise ValueError("Both rollout buffers must contain data before update().")

        mean_cost, new_lambda = self.update_dual(mlb_buffer.costs)
        adv_stats = self._prepare_targets(
            mlb_buffer,
            es_buffer,
            next_mlb_value=next_mlb_value,
            next_cost_value=next_cost_value,
            next_es_value=next_es_value,
        )
        mlb_stats = self._update_mlb(mlb_buffer)
        es_stats = self._update_es(es_buffer)

        return {
            **mlb_stats,
            **es_stats,
            **adv_stats,
            "mean_constraint_cost": mean_cost,
            "cost_ema": self.cost_ema,
            "lagrange_lambda": new_lambda,
        }

    def save_checkpoint(self, path: str | Path, iteration: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": int(iteration),
                "es_actor": self.es_actor.state_dict(),
                "mlb_actor": self.mlb_actor.state_dict(),
                "es_critic": self.es_critic.state_dict(),
                "mlb_critic": self.mlb_critic.state_dict(),
                "cost_critic": self.cost_critic.state_dict(),
                "es_actor_opt": self.es_actor_opt.state_dict(),
                "mlb_actor_opt": self.mlb_actor_opt.state_dict(),
                "es_critic_opt": self.es_critic_opt.state_dict(),
                "mlb_critic_opt": self.mlb_critic_opt.state_dict(),
                "cost_critic_opt": self.cost_critic_opt.state_dict(),
                "lagrange_lambda": self.lagrange_lambda,
                "cost_ema": self.cost_ema,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.es_actor.load_state_dict(ckpt["es_actor"])
        self.mlb_actor.load_state_dict(ckpt["mlb_actor"])
        self.es_critic.load_state_dict(ckpt["es_critic"])
        self.mlb_critic.load_state_dict(ckpt["mlb_critic"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
        self.es_actor_opt.load_state_dict(ckpt["es_actor_opt"])
        self.mlb_actor_opt.load_state_dict(ckpt["mlb_actor_opt"])
        self.es_critic_opt.load_state_dict(ckpt["es_critic_opt"])
        self.mlb_critic_opt.load_state_dict(ckpt["mlb_critic_opt"])
        self.cost_critic_opt.load_state_dict(ckpt["cost_critic_opt"])
        self.lagrange_lambda = float(ckpt.get("lagrange_lambda", 0.0))
        self.cost_ema = float(ckpt.get("cost_ema", 0.0))
        return int(ckpt.get("iteration", 0))
